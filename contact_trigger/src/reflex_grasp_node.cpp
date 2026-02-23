// =============================================================================
// reflex_grasp_node.cpp  (contact_trigger 패키지)
//
// ── 전체 흐름 ────────────────────────────────────────────────────────────────
//
//  [SETTLING]
//    → "ready" 발행
//    → 손이 ready 포즈로 이동 + 안정화 (4초)
//
//  [READY_MONITORING]
//    → 현재각을 기준각(ready_pos_)으로 저장
//    → 100Hz 마다 Σ|Δθ| 계산
//    → Σ|Δθ| > contact_delta_threshold 손가락 수 >= min_finger_triggers
//    → GRASPING 전이
//
//  [GRASPING]
//    → "grasp_3" 발행 (BHand 내장 동작 그대로 사용)
//    → grasp_3 포즈 완료 대기
//      (현재각 - grasp_3 목표각 오차 < grasp_reach_threshold)
//    → 완료되면 SNN_STABILIZING 전이
//
//  [SNN_STABILIZING]
//    → 토크(effort) 감시
//    → effort > effort_contact_threshold 손가락 수 >= min_effort_fingers
//      → 물체 부하 확인 → SNN slip 보정 시작
//    → slip = max(0, grasp_target_MCP - cur_MCP)
//    → LIF 뉴런 적분 → spike
//    → grasp_target MCP/PIP/DIP += slip_target_increment
//    → "pdControl" + joint_cmd 로 더 세게 잡기
//
//  [리셋]
//    → 키보드 'r' → SETTLING
// =============================================================================

#include "contact_trigger/reflex_grasp_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace contact_trigger
{

// =============================================================================
// grasp_3 실측 목표각 [rad]
// =============================================================================
static constexpr double GRASP3[16] = {
    //  spread      MCP        PIP        DIP
       -0.03499,   1.28901,   0.92625,   0.58213,   // finger0 (index)
        0.15428,   1.27770,   0.91272,   0.66263,   // finger1 (middle)
        0.05586,   0.46077,   0.59770,   0.75628,   // finger2 (ring)
    //  rotation    MCP        PIP        DIP
        1.40778,   0.06167,  -0.24653,   1.78327    // finger3 (thumb)
};

// slip 보상 방향 (+1 = 증가, -1 = 감소)
// thumb PIP 는 음수 방향이 더 닫히는 방향
static constexpr double SLIP_DIR[N_FIN][4] = {
    { 0.0,  1.0,  1.0,  1.0 },   // finger0
    { 0.0,  1.0,  1.0,  1.0 },   // finger1
    { 0.0,  1.0,  1.0,  1.0 },   // finger2
    { 0.0,  1.0, -1.0,  1.0 }    // finger3 (thumb PIP 반대)
};

// slip 보상 한계값
static constexpr double SLIP_LIMIT[N_FIN][4] = {
    { 0.0,  1.57,  1.57,  1.57 },
    { 0.0,  1.57,  1.57,  1.57 },
    { 0.0,  1.57,  1.57,  1.57 },
    { 0.0,  1.57, -0.50,  1.57 }   // thumb PIP 하한
};

// =============================================================================
// Constructor
// =============================================================================
ReflexGraspNode::ReflexGraspNode(const rclcpp::NodeOptions & options)
: Node("reflex_grasp_node", options)
{
    declare_parameter("contact_delta_threshold",   0.05);
    declare_parameter("min_finger_triggers",       1);
    declare_parameter("grasp_reach_threshold",     0.15);
    declare_parameter("effort_contact_threshold",  0.3);
    declare_parameter("min_effort_fingers",        1);
    declare_parameter("slip_threshold",            0.001);
    declare_parameter("snn_tau_slip",              0.03);
    declare_parameter("slip_target_increment",     0.05);
    declare_parameter("slip_target_max",           1.57);
    declare_parameter("settle_frames",             400);
    declare_parameter("grasp_hold_sec",            0.0);

    contact_delta_threshold_  = get_parameter("contact_delta_threshold").as_double();
    min_finger_triggers_      = get_parameter("min_finger_triggers").as_int();
    grasp_reach_threshold_    = get_parameter("grasp_reach_threshold").as_double();
    effort_contact_threshold_ = get_parameter("effort_contact_threshold").as_double();
    min_effort_fingers_       = get_parameter("min_effort_fingers").as_int();
    slip_threshold_           = get_parameter("slip_threshold").as_double();
    snn_tau_slip_             = get_parameter("snn_tau_slip").as_double();
    slip_target_increment_    = get_parameter("slip_target_increment").as_double();
    slip_target_max_          = get_parameter("slip_target_max").as_double();
    settle_frames_            = get_parameter("settle_frames").as_int();
    grasp_hold_sec_           = get_parameter("grasp_hold_sec").as_double();

    for (int f = 0; f < N_FIN; f++) {
        slip_neurons_[f].threshold = slip_threshold_;
        slip_neurons_[f].tau       = snn_tau_slip_;
        slip_neurons_[f].reset();
    }

    cur_effort_.fill(0.0);
    initGraspTarget();

    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_states", 10,
        std::bind(&ReflexGraspNode::onJointState, this, std::placeholders::_1));

    cmd_sub_ = create_subscription<std_msgs::msg::String>(
        "/contact_trigger/cmd", 1,
        std::bind(&ReflexGraspNode::onExtCmd, this, std::placeholders::_1));

    lib_sub_ = create_subscription<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 10,
        std::bind(&ReflexGraspNode::onLibCmd, this, std::placeholders::_1));

    lib_pub_  = create_publisher<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 1);

    jcmd_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_cmd", 1);

    dbg_pub_  = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/contact_trigger/debug", 10);

    timer_ = create_wall_timer(10ms,
        std::bind(&ReflexGraspNode::timerCb, this));

    RCLCPP_INFO(get_logger(), "==============================================");
    RCLCPP_INFO(get_logger(), "  Neuromorphic Reflex Grasp Node             ");
    RCLCPP_INFO(get_logger(), "  [접촉감지] Δθ>%.3f  min_fin=%d",
                contact_delta_threshold_, min_finger_triggers_);
    RCLCPP_INFO(get_logger(), "  [부하감지] effort>%.2f  min_fin=%d",
                effort_contact_threshold_, min_effort_fingers_);
    RCLCPP_INFO(get_logger(), "  [SNN slip] thr=%.4f  tau=%.3fs  incr=%.3f",
                slip_threshold_, snn_tau_slip_, slip_target_increment_);
    RCLCPP_INFO(get_logger(), "==============================================");

    transitionTo(ReflexState::SETTLING);
}

// =============================================================================
// initGraspTarget
// =============================================================================
void ReflexGraspNode::initGraspTarget()
{
    for (int i = 0; i < DOF; i++) grasp_target_[i] = GRASP3[i];
}

// =============================================================================
// onJointState
// =============================================================================
void ReflexGraspNode::onJointState(const sensor_msgs::msg::JointState & msg)
{
    if (static_cast<int>(msg.position.size()) < DOF) return;

    prev_pos_ = cur_pos_;
    for (int i = 0; i < DOF; i++) cur_pos_[i] = msg.position[i];

    // effort 저장
    if (static_cast<int>(msg.effort.size()) >= DOF) {
        for (int i = 0; i < DOF; i++) cur_effort_[i] = msg.effort[i];
    }

    js_received_ = true;
}

// =============================================================================
// onLibCmd
// =============================================================================
void ReflexGraspNode::onLibCmd(const std_msgs::msg::String & msg)
{
    if (msg.data != "ready") return;
    if (ignore_next_ready_) {
        ignore_next_ready_ = false;
        return;
    }
    RCLCPP_INFO(get_logger(), "키보드 'r' 감지 → SETTLING 리셋");
    transitionTo(ReflexState::SETTLING);
}

// =============================================================================
// onExtCmd
// =============================================================================
void ReflexGraspNode::onExtCmd(const std_msgs::msg::String & msg)
{
    const auto & cmd = msg.data;
    if      (cmd == "reset" || cmd == "ready") transitionTo(ReflexState::SETTLING);
    else if (cmd == "grasp")  transitionTo(ReflexState::GRASPING);
    else if (cmd == "off") { sendLibCmd("off"); transitionTo(ReflexState::IDLE); }
    else if (cmd == "status")
        RCLCPP_INFO(get_logger(), "상태: %s", stateName(state_).c_str());
}

// =============================================================================
// timerCb (100Hz)
// =============================================================================
void ReflexGraspNode::timerCb()
{
    if (!js_received_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                             "joint_states 대기 중...");
        return;
    }
    switch (state_) {
        case ReflexState::SETTLING:         tickSettling();        break;
        case ReflexState::READY_MONITORING: tickMonitoring();      break;
        case ReflexState::GRASPING:         tickGrasping();        break;
        case ReflexState::SNN_STABILIZING:  tickSnnStabilizing();  break;
        case ReflexState::IDLE:             break;
    }
}

// =============================================================================
// tickSettling
// =============================================================================
void ReflexGraspNode::tickSettling()
{
    frame_count_++;
    if (frame_count_ >= settle_frames_ && isStable()) {
        transitionTo(ReflexState::READY_MONITORING);
        return;
    }
    if (frame_count_ >= settle_frames_ * 3) {
        RCLCPP_WARN(get_logger(), "안정화 타임아웃 → 강제 READY_MONITORING");
        transitionTo(ReflexState::READY_MONITORING);
    }
}

// =============================================================================
// tickMonitoring
// 단순 Δθ 비교 → 물체 접촉 감지 → GRASPING
// =============================================================================
void ReflexGraspNode::tickMonitoring()
{
    int triggered = 0;
    for (int f = 0; f < N_FIN; f++) {
        double delta = 0.0;
        for (int j = 0; j < 4; j++)
            delta += std::abs(cur_pos_[f*4+j] - ready_pos_[f*4+j]);
        if (delta > contact_delta_threshold_) triggered++;
    }

    publishDebug();

    if (triggered >= min_finger_triggers_) {
        RCLCPP_INFO(get_logger(),
                    "★ 접촉 감지! %d개 손가락 Δθ 초과 → GRASPING", triggered);
        transitionTo(ReflexState::GRASPING);
    }
}

// =============================================================================
// tickGrasping
// grasp_3 포즈 완료 대기
// 현재각이 grasp_3 목표각에 충분히 가까워지면 → SNN_STABILIZING
// =============================================================================
void ReflexGraspNode::tickGrasping()
{
    publishDebug();

    if (isGraspReached()) {
        RCLCPP_INFO(get_logger(),
                    "★ grasp_3 포즈 완료 → SNN_STABILIZING");
        transitionTo(ReflexState::SNN_STABILIZING);
    }

    // 타임아웃 (5초)
    if ((now() - grasp_start_time_).seconds() > 5.0) {
        RCLCPP_WARN(get_logger(), "grasp_3 타임아웃 → 강제 SNN_STABILIZING");
        transitionTo(ReflexState::SNN_STABILIZING);
    }
}

// =============================================================================
// tickSnnStabilizing
//
// [동작 원리]
//  1. effort 감시: 물체 부하가 실제로 걸려있는지 확인
//     → effort_contact_threshold 이상인 손가락 수 >= min_effort_fingers
//     → 부하 있음 → SNN slip 보정 동작
//
//  2. SNN slip 보정:
//     slip = max(0, grasp_target_MCP - cur_MCP)
//     → LIF 뉴런 적분 → spike
//     → grasp_target += slip_target_increment
//     → joint_cmd 로 더 깊은 목표각 전달
//     → PD 토크 증가 → 더 세게 잡음
// =============================================================================
void ReflexGraspNode::tickSnnStabilizing()
{
    // 부하 감지
    if (isEffortDetected()) {
        runSlipSNN();
    }

    publishDebug();

    // grasp_hold_sec 초과 시 리셋
    if (grasp_hold_sec_ > 0.0 &&
        (now() - grasp_start_time_).seconds() > grasp_hold_sec_) {
        RCLCPP_INFO(get_logger(), "유지 시간 초과 → SETTLING");
        transitionTo(ReflexState::SETTLING);
    }
}

// =============================================================================
// isGraspReached
// 손가락 4개 중 3개 이상이 grasp_3 목표각 근처에 도달했으면 완료
// =============================================================================
bool ReflexGraspNode::isGraspReached() const
{
    int reached = 0;
    for (int f = 0; f < N_FIN; f++) {
        double err = 0.0;
        for (int j = 0; j < 4; j++)
            err += std::abs(cur_pos_[f*4+j] - GRASP3[f*4+j]);
        if (err < grasp_reach_threshold_) reached++;
    }
    return reached >= 3;
}

// =============================================================================
// isEffortDetected
// MCP 관절 토크가 threshold 이상인 손가락 수 확인
// =============================================================================
bool ReflexGraspNode::isEffortDetected() const
{
    int count = 0;
    for (int f = 0; f < N_FIN; f++) {
        int mcp = f * 4 + 1;
        if (std::abs(cur_effort_[mcp]) > effort_contact_threshold_) count++;
    }
    return count >= min_effort_fingers_;
}

// =============================================================================
// runSlipSNN
//
// [논문 연결] FeFET-PIM Crossbar SNN slip reflex
//
//  입력 전류 I = max(0, grasp_target_MCP - cur_MCP)
//    → grasp_3 목표각보다 실제 관절이 열려있는 정도 = slip 정도
//  LIF 뉴런 적분 → threshold 초과 → spike
//  spike 시:
//    ① grasp_target MCP/PIP/DIP += slip_target_increment (방향 고려)
//    ② joint_cmd 발행 → PD 토크 증가 → 더 세게 잡음
// =============================================================================
void ReflexGraspNode::runSlipSNN()
{
    bool any_spike = false;

    for (int f = 0; f < N_FIN; f++) {
        int    mcp  = f * 4 + 1;
        double slip = std::max(0.0, grasp_target_[mcp] - cur_pos_[mcp]);

        if (slip_neurons_[f].update(slip, DT)) {
            slip_event_count_++;
            any_spike = true;

            RCLCPP_WARN(get_logger(),
                        "★ Slip SNN spike! finger%d  slip=%.4f rad  "
                        "→ target +%.3f  (events=%d)",
                        f, slip, slip_target_increment_, slip_event_count_);

            // 방향 고려해서 목표각 증가
            for (int j = 1; j < 4; j++) {
                int    idx = f * 4 + j;
                double dir = SLIP_DIR[f][j];
                double lim = SLIP_LIMIT[f][j];

                if (dir > 0.0) {
                    grasp_target_[idx] = std::min(
                        grasp_target_[idx] + slip_target_increment_, lim);
                } else {
                    grasp_target_[idx] = std::max(
                        grasp_target_[idx] - slip_target_increment_, lim);
                }
            }

            RCLCPP_INFO(get_logger(),
                        "  finger%d new target → MCP=%.3f PIP=%.3f DIP=%.3f",
                        f,
                        grasp_target_[f*4+1],
                        grasp_target_[f*4+2],
                        grasp_target_[f*4+3]);
        }
    }

    // spike 발생 시 새 목표각으로 joint_cmd 발행
    if (any_spike) {
        cmd_pos_ = grasp_target_;
        sendJointCmd();
    }
}

// =============================================================================
// transitionTo
// =============================================================================
void ReflexGraspNode::transitionTo(ReflexState next)
{
    RCLCPP_INFO(get_logger(), "[전이] %s → %s",
                stateName(state_).c_str(), stateName(next).c_str());
    state_ = next;

    switch (next)
    {
    case ReflexState::SETTLING:
        ignore_next_ready_ = true;
        sendLibCmd("ready");
        frame_count_      = 0;
        slip_event_count_ = 0;
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        initGraspTarget();
        RCLCPP_INFO(get_logger(), "ready 발행 → %.1fs 안정화 대기",
                    settle_frames_ / 100.0);
        break;

    case ReflexState::READY_MONITORING:
        ready_pos_ = cur_pos_;
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        RCLCPP_INFO(get_logger(), "★ 기준각 저장 완료                       ★");
        RCLCPP_INFO(get_logger(), "★ 물체를 손가락에 살짝 갖다 대세요       ★");
        RCLCPP_INFO(get_logger(), "  Δθ > %.3f rad 감지 시 grasp_3 시작",
                    contact_delta_threshold_);
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        break;

    case ReflexState::GRASPING:
        // BHand 내장 grasp_3 동작 직접 호출
        sendLibCmd("grasp_3");
        grasp_start_time_ = now();
        RCLCPP_INFO(get_logger(), "★★★ grasp_3 시작! ★★★");
        RCLCPP_INFO(get_logger(), "  포즈 완료 대기 중...");
        break;

    case ReflexState::SNN_STABILIZING:
        // pdControl 전환 후 joint_cmd 로 slip 보정
        sendLibCmd("pdControl");
        cmd_pos_ = cur_pos_;
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        grasp_start_time_ = now();
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        RCLCPP_INFO(get_logger(), "★ SNN slip 보정 시작                     ★");
        RCLCPP_INFO(get_logger(), "  effort > %.2f 감지 시 SNN 동작",
                    effort_contact_threshold_);
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        break;

    case ReflexState::IDLE:
        RCLCPP_INFO(get_logger(), "IDLE");
        break;
    }
}

// =============================================================================
// 유틸
// =============================================================================
bool ReflexGraspNode::isStable() const
{
    for (int i = 0; i < DOF; i++)
        if (std::abs(cur_pos_[i] - prev_pos_[i]) > 0.003) return false;
    return true;
}

void ReflexGraspNode::sendLibCmd(const std::string & cmd)
{
    auto msg = std_msgs::msg::String();
    msg.data = cmd;
    lib_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "→ lib_cmd: [%s]", cmd.c_str());
}

void ReflexGraspNode::sendJointCmd()
{
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.position.resize(DOF);
    for (int i = 0; i < DOF; i++) js.position[i] = cmd_pos_[i];
    jcmd_pub_->publish(js);
}

// /contact_trigger/debug 레이아웃:
//  [0~3]  손가락별 Σ|Δθ|        (접촉 감지값, threshold=0.05)
//  [4~7]  손가락별 MCP effort    (부하 감지값)
//  [8~11] slip 뉴런 막전위       (threshold 넘으면 spike)
//  [12]   slip_event_count_
//  [13]   현재 상태 (0~4)
void ReflexGraspNode::publishDebug()
{
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(14);

    for (int f = 0; f < N_FIN; f++) {
        double delta = 0.0;
        for (int j = 0; j < 4; j++)
            delta += std::abs(cur_pos_[f*4+j] - ready_pos_[f*4+j]);
        msg.data[f]   = delta;
        msg.data[f+4] = cur_effort_[f*4+1];   // MCP effort
        msg.data[f+8] = slip_neurons_[f].membrane;
    }
    msg.data[12] = static_cast<double>(slip_event_count_);
    msg.data[13] = static_cast<double>(state_);

    dbg_pub_->publish(msg);
}

std::string ReflexGraspNode::stateName(ReflexState s) const
{
    switch (s) {
        case ReflexState::IDLE:             return "IDLE";
        case ReflexState::SETTLING:         return "SETTLING";
        case ReflexState::READY_MONITORING: return "READY_MONITORING";
        case ReflexState::GRASPING:         return "GRASPING";
        case ReflexState::SNN_STABILIZING:  return "SNN_STABILIZING";
        default:                            return "UNKNOWN";
    }
}

}  // namespace contact_trigger

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<contact_trigger::ReflexGraspNode>());
    rclcpp::shutdown();
    return 0;
}
