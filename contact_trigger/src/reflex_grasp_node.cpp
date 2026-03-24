// =============================================================================
// reflex_grasp_node.cpp
//
// ── allegro_node_grasp.cpp 에서 확인한 핵심 사실 ──────────────────────────────
//
//  void AllegroNodeGrasp::setJointCallback(JointState msg) {
//      desired_position = msg.position
//      pBHand->SetJointDesiredPosition(desired_position)
//      pBHand->SetMotionType(eMotionType_JOINT_PD)   ← 자동 PD 모드 전환
//  }
//
//  → joint_cmd 토픽만 보내면 자동으로 PD 제어 모드로 전환됨
//  → "pdControl" 명령 따로 안 보내도 됨
//  → joint_cmd 를 매 tick 보간해서 보내면 천천히 닫기 완성
//
//  envelop_torque 는 eMotionType_ENVELOP 전용 → grasp_3 에 효과 없음
//  slip 대응: grasp_target_ 굽힘 관절을 slip_push_ 만큼 추가로 닫기
// =============================================================================

#include "contact_trigger/reflex_grasp_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace contact_trigger
{

ReflexGraspNode::ReflexGraspNode(const rclcpp::NodeOptions & options)
: Node("reflex_grasp_node", options),
  state_(ReflexState::IDLE),
  frame_count_(0),
  joint_state_received_(false),
  ignore_next_ready_(false),
  calib_stable_count_(0),
  grasp_target_learned_(false),
  slip_event_count_(0)
{
    this->declare_parameter("contact_threshold",   0.012);
    this->declare_parameter("snn_tau_contact",     0.05);
    this->declare_parameter("min_finger_triggers", 1);
    this->declare_parameter("snn_tau_slip",        0.03);
    this->declare_parameter("slip_threshold",      0.001);
    // 닫기 속도 [rad/s]: 0.05=14s, 0.10=7s, 0.20=3.5s, 0.50=1.4s
    this->declare_parameter("grasp_speed",         0.20);
    // slip 1회 감지 시 목표 관절각 추가로 닫는 양 [rad]
    this->declare_parameter("slip_push",           0.02);
    this->declare_parameter("settle_frames",       300);
    this->declare_parameter("grasp_hold_sec",      0.0);

    contact_threshold_   = this->get_parameter("contact_threshold").as_double();
    snn_tau_contact_     = this->get_parameter("snn_tau_contact").as_double();
    min_finger_triggers_ = this->get_parameter("min_finger_triggers").as_int();
    snn_tau_slip_        = this->get_parameter("snn_tau_slip").as_double();
    slip_threshold_      = this->get_parameter("slip_threshold").as_double();
    grasp_speed_         = this->get_parameter("grasp_speed").as_double();
    slip_push_           = this->get_parameter("slip_push").as_double();
    settle_frames_       = this->get_parameter("settle_frames").as_int();
    grasp_hold_sec_      = this->get_parameter("grasp_hold_sec").as_double();

    for (int f = 0; f < N_FIN; f++) {
        contact_neurons_[f].threshold = contact_threshold_;
        contact_neurons_[f].tau       = snn_tau_contact_;
        contact_neurons_[f].reset();
        slip_neurons_[f].threshold    = slip_threshold_;
        slip_neurons_[f].tau          = snn_tau_slip_;
        slip_neurons_[f].reset();
    }

    ready_position_.fill(0.0);
    current_position_.fill(0.0);
    prev_position_.fill(0.0);
    cmd_position_.fill(0.0);
    grasp_target_.fill(0.0);

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_states", 10,
        std::bind(&ReflexGraspNode::jointStateCallback, this, std::placeholders::_1));

    cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/contact_trigger/cmd", 1,
        std::bind(&ReflexGraspNode::externalCmdCallback, this, std::placeholders::_1));

    allegro_lib_cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 10,
        std::bind(&ReflexGraspNode::allegroLibCmdCallback, this, std::placeholders::_1));

    lib_cmd_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 1);

    joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_cmd", 1);

    debug_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/contact_trigger/debug", 10);

    timer_ = this->create_wall_timer(
        10ms, std::bind(&ReflexGraspNode::timerLoop, this));

    RCLCPP_INFO(this->get_logger(), "=============================================");
    RCLCPP_INFO(this->get_logger(), "  Neuromorphic Reflex Grasp Node            ");
    RCLCPP_INFO(this->get_logger(), "  joint_cmd → 자동 PD 모드 (pdControl 불필요)");
    RCLCPP_INFO(this->get_logger(), "  [접촉SNN] threshold=%.4f tau=%.3fs",
                contact_threshold_, snn_tau_contact_);
    RCLCPP_INFO(this->get_logger(), "  [slip SNN] threshold=%.4f tau=%.3fs",
                slip_threshold_, snn_tau_slip_);
    RCLCPP_INFO(this->get_logger(), "  [속도] %.2f rad/s  [slip대응] %.3f rad/spike",
                grasp_speed_, slip_push_);
    RCLCPP_INFO(this->get_logger(), "=============================================");

    transitionTo(ReflexState::CALIBRATING);
}

// =============================================================================
// tickCalibrating
// grasp_3 실행 후 손이 완전히 멈추면 그 관절각을 grasp_target_ 으로 저장
// =============================================================================
void ReflexGraspNode::tickCalibrating()
{
    frame_count_++;
    if (frame_count_ < 200) return;  // 최소 2초 대기

    if (isHandStable()) calib_stable_count_++;
    else                calib_stable_count_ = 0;

    if (calib_stable_count_ >= 50) {  // 0.5초 연속 안정 → 학습 완료
        grasp_target_         = current_position_;
        grasp_target_learned_ = true;

        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        RCLCPP_INFO(this->get_logger(), "★ grasp_3 목표각 학습 완료 ★");
        for (int f = 0; f < N_FIN; f++) {
            RCLCPP_INFO(this->get_logger(),
                "  finger%d: spread=%.3f MCP=%.3f PIP=%.3f DIP=%.3f",
                f, grasp_target_[f*4], grasp_target_[f*4+1],
                   grasp_target_[f*4+2], grasp_target_[f*4+3]);
        }
        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        transitionTo(ReflexState::SETTLING);
        return;
    }

    if (frame_count_ > 1000) {  // 10초 타임아웃
        RCLCPP_WARN(this->get_logger(), "캘리브 타임아웃 → 현재 관절각으로 대체");
        grasp_target_         = current_position_;
        grasp_target_learned_ = true;
        transitionTo(ReflexState::SETTLING);
    }
}

// =============================================================================
// tickSettling
// =============================================================================
void ReflexGraspNode::tickSettling()
{
    frame_count_++;
    if (frame_count_ >= settle_frames_ && isHandStable()) {
        transitionTo(ReflexState::READY_MONITORING);
        return;
    }
    if (frame_count_ >= settle_frames_ * 3) {
        RCLCPP_WARN(this->get_logger(), "안정화 타임아웃 → 강제 READY_MONITORING");
        transitionTo(ReflexState::READY_MONITORING);
    }
}

// =============================================================================
// tickMonitoring
// =============================================================================
void ReflexGraspNode::tickMonitoring()
{
    int fired = runContactSNN();
    publishDebug();
    if (fired >= min_finger_triggers_) {
        RCLCPP_INFO(this->get_logger(), "SNN spike %d개 → grasp 시작", fired);
        transitionTo(ReflexState::GRASPING);
    }
}

// =============================================================================
// tickGrasping
//
// 매 tick(10ms):
//   ① cmd_position_ → grasp_target_ 방향으로 grasp_speed * DT 만큼 이동
//   ② joint_cmd 발행 → allegro_node_grasp 가 자동으로 PD 모드 전환
//   ③ slip SNN 실행
//      spike → grasp_target_ 굽힘 관절 += slip_push_ (더 조임)
// =============================================================================
void ReflexGraspNode::tickGrasping()
{
    // ① 보간
    const double step = grasp_speed_ * DT;
    for (int i = 0; i < DOF; i++) {
        double diff = grasp_target_[i] - cmd_position_[i];
        if (std::abs(diff) > step)
            cmd_position_[i] += (diff > 0.0) ? step : -step;
        else
            cmd_position_[i] = grasp_target_[i];
    }

    // ② joint_cmd 발행
    sendJointCmd();

    // ③ slip SNN
    if (runSlipSNN()) {
        // finger0, finger1 굽힘(j=1~3) 만 더 닫기
        // finger2(ring) 는 grasp_3 미사용 → 건드리지 않음
        for (int f = 0; f < 2; f++) {
            for (int j = 1; j <= 3; j++) {
                grasp_target_[f*4+j] = std::min(
                    grasp_target_[f*4+j] + slip_push_, 1.80);
            }
        }
        // 엄지는 전체
        for (int j = 0; j < 4; j++) {
            grasp_target_[12+j] = std::min(
                grasp_target_[12+j] + slip_push_, 1.80);
        }
        RCLCPP_INFO(this->get_logger(),
                    "Slip → target ↑ %.3f rad (MCP0=%.3f MCP1=%.3f)",
                    slip_push_, grasp_target_[1], grasp_target_[5]);
    }

    publishDebug();

    if (grasp_hold_sec_ > 0.0 &&
        (this->now() - grasp_start_time_).seconds() > grasp_hold_sec_) {
        transitionTo(ReflexState::SETTLING);
    }
}

// =============================================================================
// runContactSNN
// =============================================================================
int ReflexGraspNode::runContactSNN()
{
    int fired = 0;
    for (int f = 0; f < N_FIN; f++) {
        double I = 0.0;
        for (int j = 0; j < 4; j++)
            I += std::abs(current_position_[f*4+j] - ready_position_[f*4+j]);
        if (contact_neurons_[f].update(I, DT)) fired++;
    }
    return fired;
}

// =============================================================================
// runSlipSNN
// MCP 관절이 열리는 방향으로 drift → slip 판정
// =============================================================================
bool ReflexGraspNode::runSlipSNN()
{
    bool any_spike = false;
    for (int f = 0; f < N_FIN; f++) {
        double delta    = current_position_[f*4+1] - prev_position_[f*4+1];
        double slip_vel = std::max(0.0, -delta / DT);
        if (slip_neurons_[f].update(slip_vel, DT)) {
            slip_event_count_++;
            any_spike = true;
        }
    }
    return any_spike;
}

// =============================================================================
// sendJointCmd
// =============================================================================
void ReflexGraspNode::sendJointCmd()
{
    sensor_msgs::msg::JointState js;
    js.header.stamp = this->now();
    js.position.assign(cmd_position_.begin(), cmd_position_.end());
    joint_cmd_pub_->publish(js);
}

void ReflexGraspNode::jointStateCallback(const sensor_msgs::msg::JointState & msg)
{
    if ((int)msg.position.size() < DOF) return;
    prev_position_ = current_position_;
    for (int i = 0; i < DOF; i++) current_position_[i] = msg.position[i];
    joint_state_received_ = true;
}

void ReflexGraspNode::allegroLibCmdCallback(const std_msgs::msg::String & msg)
{
    if (msg.data != "ready") return;
    if (ignore_next_ready_) { ignore_next_ready_ = false; return; }
    RCLCPP_INFO(this->get_logger(), "★ 키보드 'r' → SETTLING 리셋 ★");
    transitionTo(ReflexState::SETTLING);
}

void ReflexGraspNode::externalCmdCallback(const std_msgs::msg::String & msg)
{
    const std::string cmd = msg.data;
    if      (cmd == "reset" || cmd == "ready") transitionTo(ReflexState::SETTLING);
    else if (cmd == "grasp")  transitionTo(ReflexState::GRASPING);
    else if (cmd == "calib")  transitionTo(ReflexState::CALIBRATING);
    else if (cmd == "off")  { sendCmd("off"); transitionTo(ReflexState::IDLE); }
    else if (cmd == "status") RCLCPP_INFO(this->get_logger(), "상태: %s",
                                           stateName(state_).c_str());
    else RCLCPP_WARN(this->get_logger(), "알 수 없는 명령: %s", cmd.c_str());
}

void ReflexGraspNode::timerLoop()
{
    if (!joint_state_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "joint_states 대기 중...");
        return;
    }
    switch (state_) {
        case ReflexState::CALIBRATING:      tickCalibrating();  break;
        case ReflexState::SETTLING:         tickSettling();     break;
        case ReflexState::READY_MONITORING: tickMonitoring();   break;
        case ReflexState::GRASPING:         tickGrasping();     break;
        case ReflexState::IDLE:             break;
    }
}

bool ReflexGraspNode::isHandStable()
{
    for (int i = 0; i < DOF; i++)
        if (std::abs(current_position_[i] - prev_position_[i]) > 0.003) return false;
    return true;
}

void ReflexGraspNode::publishDebug()
{
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(13);
    for (int f = 0; f < N_FIN; f++) {
        msg.data[f]     = contact_neurons_[f].membrane;
        msg.data[f + 4] = slip_neurons_[f].membrane;
        double err = 0.0;
        for (int j = 0; j < 4; j++)
            err += std::abs(current_position_[f*4+j] - ready_position_[f*4+j]);
        msg.data[f + 8] = err;
    }
    msg.data[12] = static_cast<double>(slip_event_count_);
    debug_pub_->publish(msg);
}

void ReflexGraspNode::transitionTo(ReflexState next)
{
    RCLCPP_INFO(this->get_logger(), "[상태전이] %s → %s",
                stateName(state_).c_str(), stateName(next).c_str());
    state_ = next;

    switch (next)
    {
    case ReflexState::CALIBRATING:
        sendCmd("grasp_3");
        frame_count_ = 0; calib_stable_count_ = 0;
        RCLCPP_INFO(this->get_logger(), "grasp_3 실행 → 목표각 학습 중...");
        break;

    case ReflexState::SETTLING:
        ignore_next_ready_ = true;
        sendCmd("ready");
        frame_count_ = 0; slip_event_count_ = 0;
        for (int f = 0; f < N_FIN; f++) {
            contact_neurons_[f].reset();
            slip_neurons_[f].reset();
        }
        RCLCPP_INFO(this->get_logger(),
                    "ready 복귀 → %.1fs 안정화 대기", settle_frames_ / 100.0);
        break;

    case ReflexState::READY_MONITORING:
        ready_position_ = current_position_;
        for (int f = 0; f < N_FIN; f++) contact_neurons_[f].reset();
        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────");
        RCLCPP_INFO(this->get_logger(), "★ 기준각 저장 - 손가락을 건드세요 ★");
        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────");
        break;

    case ReflexState::GRASPING:
        // joint_cmd 보내는 순간 allegro_node_grasp 가 자동으로 PD 모드 전환
        // → pdControl 명령 불필요
        cmd_position_ = current_position_;
        // spread(j=0): finger0~2 ready 포즈값 고정 (옆으로 안 쏠림)
        for (int f = 0; f < 3; f++)
            grasp_target_[f*4] = ready_position_[f*4];
        // finger2(ring): grasp_3 에서 안 쓰임 → 굽힘 관절 전체 ready 고정
        for (int j = 1; j <= 3; j++)
            grasp_target_[2*4+j] = ready_position_[2*4+j];
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        grasp_start_time_ = this->now();
        RCLCPP_INFO(this->get_logger(),
                    "★★★ 천천히 grasp 시작 (%.2f rad/s) ★★★", grasp_speed_);
        break;

    case ReflexState::IDLE:
        break;
    }
}

void ReflexGraspNode::sendCmd(const std::string & cmd)
{
    auto msg = std_msgs::msg::String();
    msg.data = cmd;
    lib_cmd_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "→ lib_cmd: [%s]", cmd.c_str());
}

std::string ReflexGraspNode::stateName(ReflexState s) const
{
    switch (s) {
        case ReflexState::CALIBRATING:      return "CALIBRATING";
        case ReflexState::IDLE:             return "IDLE";
        case ReflexState::SETTLING:         return "SETTLING";
        case ReflexState::READY_MONITORING: return "READY_MONITORING";
        case ReflexState::GRASPING:         return "GRASPING";
        default:                            return "UNKNOWN";
    }
}

} // namespace contact_trigger

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<contact_trigger::ReflexGraspNode>());
    rclcpp::shutdown();
    return 0;
}