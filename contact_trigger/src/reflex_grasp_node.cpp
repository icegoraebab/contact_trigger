// =============================================================================
// reflex_grasp_node.cpp  (contact_trigger 패키지)
//
// 무촉각 뉴로모픽 그립 노드 - slip 적응 토크 제어
//
// ── slip 적응 토크 제어 원리 ──────────────────────────────────────────────────
//
//  grasp_3 실행 중 매 tick(10ms):
//
//  slip 감지(SNN spike 발생)
//    → envelop_torque += torque_step_up   (조금 증가)
//
//  slip 없음
//    → envelop_torque *= torque_decay_rate  (서서히 감소)
//
//  → slip이 막 멈추는 지점에서 torque 가 자동 수렴
//  → 필요 이상으로 강하게 잡지 않음
//
//  envelop_torque 범위: [torque_min_, torque_max_]
//
// ── 논문 연결 ────────────────────────────────────────────────────────────────
//  slip SNN spike = FeFET crossbar 의 에러 신호
//  torque 증가    = R-STDP weight update (FeFET conductance 증가) 모사
//  torque 감소    = synaptic depression (conductance 자연 감쇠) 모사
// =============================================================================

#include "contact_trigger/reflex_grasp_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace contact_trigger
{

// =============================================================================
// Constructor
// =============================================================================
ReflexGraspNode::ReflexGraspNode(const rclcpp::NodeOptions & options)
: Node("reflex_grasp_node", options),
  state_(ReflexState::IDLE),
  frame_count_(0),
  joint_state_received_(false),
  ignore_next_ready_(false),
  slip_event_count_(0),
  envelop_torque_(0.0)
{
    // ── 파라미터 선언 ─────────────────────────────────────────────────────────

    // [SNN 접촉 감지]
    // V_ss = Σ|Δθ| × tau  (손가락 5° ≈ 0.35rad → V_ss ≈ 0.0175)
    this->declare_parameter("contact_threshold",   0.012);
    this->declare_parameter("snn_tau_contact",     0.05);
    this->declare_parameter("min_finger_triggers", 1);

    // [SNN slip 감지]
    this->declare_parameter("snn_tau_slip",        0.03);
    this->declare_parameter("slip_threshold",      0.001);

    // [envelop torque 적응]
    // torque_min: grasp_3 기본 동작 유지 최소값
    // torque_max: 손/물체 보호를 위한 상한
    // torque_step_up: slip spike 1회당 토크 증가량
    // torque_decay_rate: 매 tick 감쇠 비율 (0.99 = 1% 감소/tick)
    this->declare_parameter("torque_min",          0.3);   // grasp_3 시작 토크
    this->declare_parameter("torque_max",          0.7);   // 최대 토크 (과도 방지)
    this->declare_parameter("torque_step_up",      0.03);  // slip 1회당 +3%
    this->declare_parameter("torque_decay_rate",   0.995); // slip 없으면 0.5%/tick 감소

    // [안정화]
    this->declare_parameter("settle_frames",       300);
    this->declare_parameter("grasp_hold_sec",      0.0);

    // ── 파라미터 로드 ─────────────────────────────────────────────────────────
    contact_threshold_   = this->get_parameter("contact_threshold").as_double();
    snn_tau_contact_     = this->get_parameter("snn_tau_contact").as_double();
    min_finger_triggers_ = this->get_parameter("min_finger_triggers").as_int();
    snn_tau_slip_        = this->get_parameter("snn_tau_slip").as_double();
    slip_threshold_      = this->get_parameter("slip_threshold").as_double();
    torque_min_          = this->get_parameter("torque_min").as_double();
    torque_max_          = this->get_parameter("torque_max").as_double();
    torque_step_up_      = this->get_parameter("torque_step_up").as_double();
    torque_decay_rate_   = this->get_parameter("torque_decay_rate").as_double();
    settle_frames_       = this->get_parameter("settle_frames").as_int();
    grasp_hold_sec_      = this->get_parameter("grasp_hold_sec").as_double();

    envelop_torque_ = torque_min_;

    // ── SNN 뉴런 파라미터 ─────────────────────────────────────────────────────
    for (int f = 0; f < N_FIN; f++) {
        contact_neurons_[f].threshold = contact_threshold_;
        contact_neurons_[f].tau       = snn_tau_contact_;
        contact_neurons_[f].reset();

        slip_neurons_[f].threshold    = slip_threshold_;
        slip_neurons_[f].tau          = snn_tau_slip_;
        slip_neurons_[f].reset();
    }

    // ── 배열 초기화 ──────────────────────────────────────────────────────────
    ready_position_.fill(0.0);
    current_position_.fill(0.0);
    prev_position_.fill(0.0);

    // ── 구독 ─────────────────────────────────────────────────────────────────
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_states", 10,
        std::bind(&ReflexGraspNode::jointStateCallback, this, std::placeholders::_1));

    cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/contact_trigger/cmd", 1,
        std::bind(&ReflexGraspNode::externalCmdCallback, this, std::placeholders::_1));

    allegro_lib_cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 10,
        std::bind(&ReflexGraspNode::allegroLibCmdCallback, this, std::placeholders::_1));

    // ── 발행 ─────────────────────────────────────────────────────────────────
    lib_cmd_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 1);

    // envelop_torque: allegro_node_grasp 가 구독 (BHand SetEnvelopTorqueScalar)
    envelop_torque_pub_ = this->create_publisher<std_msgs::msg::Float32>(
        "/allegroHand_0/envelop_torque", 10);

    // [0~3]=접촉뉴런막전위, [4~7]=slip뉴런막전위,
    // [8~11]=Σ|Δθ|, [12]=envelop_torque, [13]=slip_count
    debug_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/contact_trigger/debug", 10);

    timer_ = this->create_wall_timer(
        10ms, std::bind(&ReflexGraspNode::timerLoop, this));

    // ── 시작 로그 ─────────────────────────────────────────────────────────────
    RCLCPP_INFO(this->get_logger(), "=============================================");
    RCLCPP_INFO(this->get_logger(), "  Neuromorphic Reflex Grasp Node            ");
    RCLCPP_INFO(this->get_logger(), "  [SNN 접촉] threshold=%.4f  tau=%.3fs",
                contact_threshold_, snn_tau_contact_);
    RCLCPP_INFO(this->get_logger(), "  [SNN slip]  threshold=%.4f  tau=%.3fs",
                slip_threshold_, snn_tau_slip_);
    RCLCPP_INFO(this->get_logger(), "  [토크 적응] min=%.2f  max=%.2f  step=%.3f  decay=%.4f",
                torque_min_, torque_max_, torque_step_up_, torque_decay_rate_);
    RCLCPP_INFO(this->get_logger(), "  slip → torque +%.3f  |  no slip → torque ×%.4f",
                torque_step_up_, torque_decay_rate_);
    RCLCPP_INFO(this->get_logger(), "=============================================");

    transitionTo(ReflexState::SETTLING);
}

// =============================================================================
// jointStateCallback
// =============================================================================
void ReflexGraspNode::jointStateCallback(const sensor_msgs::msg::JointState & msg)
{
    if ((int)msg.position.size() < DOF) return;
    prev_position_ = current_position_;
    for (int i = 0; i < DOF; i++) current_position_[i] = msg.position[i];
    joint_state_received_ = true;
}

// =============================================================================
// allegroLibCmdCallback: 키보드 'r' 감지 + self-echo 방지
// =============================================================================
void ReflexGraspNode::allegroLibCmdCallback(const std_msgs::msg::String & msg)
{
    if (msg.data != "ready") return;

    if (ignore_next_ready_) {
        ignore_next_ready_ = false;
        RCLCPP_DEBUG(this->get_logger(), "self-echo 'ready' 무시");
        return;
    }

    RCLCPP_INFO(this->get_logger(),
                "★ 키보드 'r' 감지 [%s] → SETTLING 리셋 ★",
                stateName(state_).c_str());
    transitionTo(ReflexState::SETTLING);
}

// =============================================================================
// externalCmdCallback
// =============================================================================
void ReflexGraspNode::externalCmdCallback(const std_msgs::msg::String & msg)
{
    const std::string cmd = msg.data;
    RCLCPP_INFO(this->get_logger(), "수동 명령: [%s]", cmd.c_str());

    if      (cmd == "reset" || cmd == "ready") transitionTo(ReflexState::SETTLING);
    else if (cmd == "grasp")  transitionTo(ReflexState::GRASPING);
    else if (cmd == "off")  { sendCmd("off"); transitionTo(ReflexState::IDLE); }
    else if (cmd == "status") RCLCPP_INFO(this->get_logger(), "상태: %s  torque=%.3f",
                                           stateName(state_).c_str(), envelop_torque_);
    else RCLCPP_WARN(this->get_logger(), "알 수 없는 명령: %s", cmd.c_str());
}

// =============================================================================
// timerLoop (100Hz)
// =============================================================================
void ReflexGraspNode::timerLoop()
{
    if (!joint_state_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "/allegroHand_0/joint_states 대기 중...");
        return;
    }

    switch (state_) {
        case ReflexState::SETTLING:         tickSettling();   break;
        case ReflexState::READY_MONITORING: tickMonitoring(); break;
        case ReflexState::GRASPING:         tickGrasping();   break;
        case ReflexState::IDLE:             break;
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
// tickMonitoring: SNN 접촉 감지
// =============================================================================
void ReflexGraspNode::tickMonitoring()
{
    int fired = runContactSNN();
    publishDebug();

    if (fired >= min_finger_triggers_) {
        RCLCPP_INFO(this->get_logger(),
                    "SNN 접촉 spike: %d개 손가락 발화 → grasp_3 실행", fired);
        transitionTo(ReflexState::GRASPING);
    }
}

// =============================================================================
// tickGrasping
// 매 tick:
//   1. slip SNN 실행 → spike 여부 확인
//   2. spike 있으면 토크 조금 올림, 없으면 서서히 내림
//   3. 갱신된 envelop_torque 발행
// =============================================================================
void ReflexGraspNode::tickGrasping()
{
    // ① slip SNN 실행
    bool slip_occurred = runSlipSNN();

    // ② 토크 적응 (slip 수렴 제어)
    updateEnvelopTorque(slip_occurred);

    // ③ envelop_torque 발행
    publishEnvelopTorque();

    publishDebug();

    // 자동 복귀
    if (grasp_hold_sec_ > 0.0) {
        if ((this->now() - grasp_start_time_).seconds() > grasp_hold_sec_) {
            RCLCPP_INFO(this->get_logger(), "grasp 유지 시간 초과 → SETTLING");
            transitionTo(ReflexState::SETTLING);
        }
    }
}

// =============================================================================
// runContactSNN
// 손가락당 LIF 뉴런에 Σ|Δθ| 를 입력 전류로 주입
// 논문: Crossbar Σ(V·G) 온칩 연산 모사
// =============================================================================
int ReflexGraspNode::runContactSNN()
{
    int fired_count = 0;
    for (int f = 0; f < N_FIN; f++) {
        double input_current = 0.0;
        for (int j = 0; j < 4; j++) {
            input_current += std::abs(current_position_[f*4+j] - ready_position_[f*4+j]);
        }
        if (contact_neurons_[f].update(input_current, DT)) {
            RCLCPP_DEBUG(this->get_logger(),
                         "접촉 SNN: finger%d SPIKE (I=%.4f)", f, input_current);
            fired_count++;
        }
    }
    return fired_count;
}

// =============================================================================
// runSlipSNN
// grasp 중 관절각이 열리는 방향으로 drift 하면 slip으로 판정
// 각 손가락 MCP 관절(j=1) 의 역방향 이동 속도를 입력 전류로 사용
//
// 반환: 이번 tick 에 하나 이상의 손가락에서 slip spike 발생 여부
// =============================================================================
bool ReflexGraspNode::runSlipSNN()
{
    bool any_spike = false;

    for (int f = 0; f < N_FIN; f++) {
        int mcp_idx = f * 4 + 1;

        // 역방향(열리는 방향) 속도만 입력 (양수 = slip 중)
        double delta    = current_position_[mcp_idx] - prev_position_[mcp_idx];
        double slip_vel = std::max(0.0, -delta / DT);

        if (slip_neurons_[f].update(slip_vel, DT)) {
            slip_event_count_++;
            RCLCPP_DEBUG(this->get_logger(),
                         "Slip SNN: finger%d SPIKE (vel=%.4f rad/s)", f, slip_vel);
            any_spike = true;
        }
    }

    return any_spike;
}

// =============================================================================
// updateEnvelopTorque
// slip 적응 토크 수렴 제어
//
//  slip 발생  → torque += torque_step_up_       (즉시 조금 올림)
//  slip 없음  → torque *= torque_decay_rate_    (서서히 내림)
//  범위 클램프: [torque_min_, torque_max_]
//
// 수렴 원리:
//   slip이 계속 나면 torque 가 올라가다가 slip이 멈추는 순간부터
//   감소하기 시작 → slip이 막 멈추는 최소 torque 로 자동 수렴
// =============================================================================
void ReflexGraspNode::updateEnvelopTorque(bool slip_occurred)
{
    if (slip_occurred) {
        // slip 감지 → 토크 즉시 증가
        envelop_torque_ += torque_step_up_;

        RCLCPP_INFO(this->get_logger(),
                    "Slip → torque ↑ %.3f → %.3f",
                    envelop_torque_ - torque_step_up_, envelop_torque_);
    } else {
        // slip 없음 → 서서히 감소 (지수 감쇠)
        envelop_torque_ *= torque_decay_rate_;
    }

    // 범위 클램프
    envelop_torque_ = std::clamp(envelop_torque_, torque_min_, torque_max_);
}

// =============================================================================
// publishEnvelopTorque
// /allegroHand_0/envelop_torque 발행
// allegro_node_grasp 의 envelopTorqueCallback 이 수신
// → pBHand->SetEnvelopTorqueScalar(torque) 호출
// =============================================================================
void ReflexGraspNode::publishEnvelopTorque()
{
    std_msgs::msg::Float32 msg;
    msg.data = static_cast<float>(envelop_torque_);
    envelop_torque_pub_->publish(msg);
}

// =============================================================================
// resetEnvelopTorque
// SETTLING 진입 시 토크를 최솟값으로 리셋
// =============================================================================
void ReflexGraspNode::resetEnvelopTorque()
{
    envelop_torque_ = torque_min_;
    publishEnvelopTorque();
}

// =============================================================================
// isHandStable
// =============================================================================
bool ReflexGraspNode::isHandStable()
{
    for (int i = 0; i < DOF; i++) {
        if (std::abs(current_position_[i] - prev_position_[i]) > 0.003) return false;
    }
    return true;
}

// =============================================================================
// publishDebug
// [0~3]=접촉뉴런막전위, [4~7]=slip뉴런막전위,
// [8~11]=Σ|Δθ|, [12]=envelop_torque, [13]=slip_count
// =============================================================================
void ReflexGraspNode::publishDebug()
{
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(14);

    for (int f = 0; f < N_FIN; f++) {
        msg.data[f]     = contact_neurons_[f].membrane;
        msg.data[f + 4] = slip_neurons_[f].membrane;

        double err = 0.0;
        for (int j = 0; j < 4; j++) {
            err += std::abs(current_position_[f*4+j] - ready_position_[f*4+j]);
        }
        msg.data[f + 8] = err;
    }
    msg.data[12] = envelop_torque_;
    msg.data[13] = static_cast<double>(slip_event_count_);

    debug_pub_->publish(msg);
}

// =============================================================================
// transitionTo
// =============================================================================
void ReflexGraspNode::transitionTo(ReflexState next)
{
    RCLCPP_INFO(this->get_logger(), "[상태전이] %s → %s",
                stateName(state_).c_str(), stateName(next).c_str());
    state_ = next;

    switch (next)
    {
    case ReflexState::SETTLING:
        ignore_next_ready_ = true;
        sendCmd("ready");
        frame_count_      = 0;
        slip_event_count_ = 0;

        // SNN 리셋
        for (int f = 0; f < N_FIN; f++) {
            contact_neurons_[f].reset();
            slip_neurons_[f].reset();
        }

        // 토크 최솟값으로 리셋
        resetEnvelopTorque();

        RCLCPP_INFO(this->get_logger(),
                    "ready 명령 발행 → %.1fs + 안정화 대기", settle_frames_ / 100.0);
        break;

    case ReflexState::READY_MONITORING:
        ready_position_ = current_position_;
        for (int f = 0; f < N_FIN; f++) contact_neurons_[f].reset();

        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        RCLCPP_INFO(this->get_logger(), "★ 기준각 저장 완료 - SNN 감지 대기 중 ★");
        RCLCPP_INFO(this->get_logger(), "  손가락을 건드세요!");
        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        break;

    case ReflexState::GRASPING:
        // BHand grasp_3 명령 + 초기 envelop torque 설정
        sendCmd("grasp_3");
        envelop_torque_ = torque_min_;
        publishEnvelopTorque();

        // slip SNN 리셋
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        grasp_start_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "★★★ SNN spike! grasp_3 + 토크 적응 시작 ★★★");
        RCLCPP_INFO(this->get_logger(), "  초기 토크=%.2f | 범위=[%.2f, %.2f]",
                    envelop_torque_, torque_min_, torque_max_);
        RCLCPP_INFO(this->get_logger(), "  slip → +%.3f | 감쇠 ×%.4f/tick",
                    torque_step_up_, torque_decay_rate_);
        break;

    case ReflexState::IDLE:
        RCLCPP_INFO(this->get_logger(), "IDLE");
        break;
    }
}

// =============================================================================
// sendCmd
// =============================================================================
void ReflexGraspNode::sendCmd(const std::string & cmd)
{
    auto msg = std_msgs::msg::String();
    msg.data = cmd;
    lib_cmd_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "→ lib_cmd: [%s]", cmd.c_str());
}

// =============================================================================
// stateName
// =============================================================================
std::string ReflexGraspNode::stateName(ReflexState s) const
{
    switch (s) {
        case ReflexState::IDLE:             return "IDLE";
        case ReflexState::SETTLING:         return "SETTLING";
        case ReflexState::READY_MONITORING: return "READY_MONITORING";
        case ReflexState::GRASPING:         return "GRASPING";
        default:                            return "UNKNOWN";
    }
}

}  // namespace contact_trigger

// =============================================================================
// main
// =============================================================================
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<contact_trigger::ReflexGraspNode>());
    rclcpp::shutdown();
    return 0;
}