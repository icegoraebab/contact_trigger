// =============================================================================
// reflex_grasp_node.cpp  (contact_trigger 패키지)
//
// ── 핵심 흐름 ────────────────────────────────────────────────────────────────
//  1. SETTLING
//     → "ready" 발행 → 손 안정화 4초 대기
//
//  2. READY_MONITORING
//     → 현재각을 기준각(ready_pos_)으로 저장
//     → 100Hz 마다 Σ|Δθ| 계산
//     → Σ|Δθ| > contact_delta_threshold 인 손가락 수 카운트
//     → min_finger_triggers 이상이면 → SLOW_GRASPING
//     (SNN 없음, 단순 임계값 비교)
//
//  3. SLOW_GRASPING
//     → "pdControl" 발행
//     → grasp_3 목표각으로 0.3 rad/s 천천히 보간 닫기
//     → 손가락이 물체에 닿으면 속도 급감 → 그 위치에서 MCP 홀드
//     → [SNN 동작 시작]
//        slip = max(0, cmd_MCP - cur_MCP)  (목표보다 실제가 열려있는 정도)
//        → LIF 뉴런 적분 → threshold 초과 → spike
//        → grasp_target MCP/PIP/DIP += slip_target_increment
//        → fin_contacted 리셋 → 더 깊이 닫기 재시작
//        → PD 토크 자연히 증가 → 더 세게 잡음
//
//  4. 키보드 'r' → SETTLING 리셋 (grasp_target 도 초기화)
// =============================================================================

#include "contact_trigger/reflex_grasp_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace contact_trigger
{

// =============================================================================
// grasp_3 초기 목표각 [rad]
// slip 발생 시 MCP/PIP/DIP 가 동적으로 증가
// =============================================================================
static constexpr double GRASP3[16] = {
    //  spread      MCP        PIP        DIP
       -0.03499,   1.28901,   0.92625,   0.58213,   // finger0 (index)
        0.15428,   1.27770,   0.91272,   0.66263,   // finger1 (middle)
        0.05586,   0.46077,   0.59770,   0.75628,   // finger2 (ring)
    //  rotation    MCP        PIP        DIP
        1.40778,   0.06167,  -0.24653,   1.78327    // finger3 (thumb)
};



// =============================================================================
// Constructor
// =============================================================================
ReflexGraspNode::ReflexGraspNode(const rclcpp::NodeOptions & options)
: Node("reflex_grasp_node", options)
{
    // ── 파라미터 선언 ────────────────────────────────────────────────────────
    declare_parameter("contact_delta_threshold",      0.05);
    declare_parameter("min_finger_triggers",          1);
    declare_parameter("slip_threshold",               0.001);
    declare_parameter("snn_tau_slip",                 0.03);
    declare_parameter("slip_target_increment",        0.05);
    declare_parameter("slip_target_max",              1.57);
    declare_parameter("grasp_speed",                  0.3);
    declare_parameter("object_contact_vel_threshold", 0.008);
    declare_parameter("settle_frames",                400);
    declare_parameter("grasp_hold_sec",               0.0);

    // ── 파라미터 로드 ────────────────────────────────────────────────────────
    contact_delta_threshold_      = get_parameter("contact_delta_threshold").as_double();
    min_finger_triggers_          = get_parameter("min_finger_triggers").as_int();
    slip_threshold_               = get_parameter("slip_threshold").as_double();
    snn_tau_slip_                 = get_parameter("snn_tau_slip").as_double();
    slip_target_increment_        = get_parameter("slip_target_increment").as_double();
    slip_target_max_              = get_parameter("slip_target_max").as_double();
    grasp_speed_                  = get_parameter("grasp_speed").as_double();
    object_contact_vel_threshold_ = get_parameter("object_contact_vel_threshold").as_double();
    settle_frames_                = get_parameter("settle_frames").as_int();
    grasp_hold_sec_               = get_parameter("grasp_hold_sec").as_double();

    // ── SNN slip 뉴런 초기화 ─────────────────────────────────────────────────
    for (int f = 0; f < N_FIN; f++) {
        slip_neurons_[f].threshold = slip_threshold_;
        slip_neurons_[f].tau       = snn_tau_slip_;
        slip_neurons_[f].reset();
    }

    fin_contacted_.fill(false);
    fin_hold_mcp_.fill(0.0);
    initGraspTarget();

    // ── 구독 ─────────────────────────────────────────────────────────────────
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_states", 10,
        std::bind(&ReflexGraspNode::onJointState, this, std::placeholders::_1));

    cmd_sub_ = create_subscription<std_msgs::msg::String>(
        "/contact_trigger/cmd", 1,
        std::bind(&ReflexGraspNode::onExtCmd, this, std::placeholders::_1));

    lib_sub_ = create_subscription<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 10,
        std::bind(&ReflexGraspNode::onLibCmd, this, std::placeholders::_1));

    // ── 발행 ─────────────────────────────────────────────────────────────────
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
    RCLCPP_INFO(get_logger(), "  [접촉감지] Δθ > %.4f rad  min_fingers=%d",
                contact_delta_threshold_, min_finger_triggers_);
    RCLCPP_INFO(get_logger(), "  [SNN slip] thr=%.4f  tau=%.3fs",
                slip_threshold_, snn_tau_slip_);
    RCLCPP_INFO(get_logger(), "  [slip보상] +%.3f rad/spike  max=%.2f rad",
                slip_target_increment_, slip_target_max_);
    RCLCPP_INFO(get_logger(), "  [속도]     %.2f rad/s  settle=%.1fs",
                grasp_speed_, settle_frames_ / 100.0);
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
    js_received_ = true;
}

// =============================================================================
// onLibCmd  — 키보드 'r' 감지 (self-echo 방지)
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
// onExtCmd  — 수동 명령
// =============================================================================
void ReflexGraspNode::onExtCmd(const std_msgs::msg::String & msg)
{
    const auto & cmd = msg.data;
    RCLCPP_INFO(get_logger(), "수동 명령: [%s]", cmd.c_str());

    if      (cmd == "reset" || cmd == "ready") transitionTo(ReflexState::SETTLING);
    else if (cmd == "grasp")  transitionTo(ReflexState::SLOW_GRASPING);
    else if (cmd == "off") {
        sendLibCmd("off");
        transitionTo(ReflexState::IDLE);
    }
    else if (cmd == "status")
        RCLCPP_INFO(get_logger(), "현재 상태: %s", stateName(state_).c_str());
    else
        RCLCPP_WARN(get_logger(), "알 수 없는 명령: %s", cmd.c_str());
}

// =============================================================================
// timerCb (100 Hz)
// =============================================================================
void ReflexGraspNode::timerCb()
{
    if (!js_received_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                             "joint_states 수신 대기 중...");
        return;
    }
    switch (state_) {
        case ReflexState::SETTLING:         tickSettling();   break;
        case ReflexState::READY_MONITORING: tickMonitoring(); break;
        case ReflexState::SLOW_GRASPING:    tickGrasping();   break;
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
//
// SNN 없음 — 단순 Δθ 임계값 비교
//   Σ|cur - ready| > contact_delta_threshold 인 손가락 수 카운트
//   → min_finger_triggers 이상이면 grasp 트리거
// =============================================================================
void ReflexGraspNode::tickMonitoring()
{
    int triggered = 0;

    for (int f = 0; f < N_FIN; f++) {
        double delta = 0.0;
        for (int j = 0; j < 4; j++)
            delta += std::abs(cur_pos_[f*4+j] - ready_pos_[f*4+j]);

        if (delta > contact_delta_threshold_)
            triggered++;
    }

    publishDebug();

    if (triggered >= min_finger_triggers_) {
        RCLCPP_INFO(get_logger(),
                    "★ 접촉 감지! %d개 손가락 Δθ 초과 → SLOW_GRASPING", triggered);
        transitionTo(ReflexState::SLOW_GRASPING);
    }
}

// =============================================================================
// tickGrasping
//
// ① SNN slip 감지 (핵심)
// ② 물체 크기 적응
// ③ 보간 닫기
// =============================================================================
void ReflexGraspNode::tickGrasping()
{
    // ① SNN slip 감지 — 물체 잡은 후 미끄러지면 target 증가
    runSlipSNN();

    // ② 물체 크기 적응 — 속도 급감 손가락 홀드
    detectObjectContact();

    // ③ 보간 닫기
    const double step = grasp_speed_ * DT;
    bool all_done = true;

    for (int f = 0; f < N_FIN; f++) {
        int base = f * 4;

        if (fin_contacted_[f]) {
            // spread → target, MCP → 홀드, PIP·DIP → target
            cmd_pos_[base + 0] = grasp_target_[base + 0];
            cmd_pos_[base + 1] = fin_hold_mcp_[f];
            cmd_pos_[base + 2] = grasp_target_[base + 2];
            cmd_pos_[base + 3] = grasp_target_[base + 3];
            continue;
        }

        for (int j = 0; j < 4; j++) {
            int    idx  = base + j;
            double diff = grasp_target_[idx] - cmd_pos_[idx];
            if (std::abs(diff) > step) {
                cmd_pos_[idx] += (diff > 0.0) ? step : -step;
                all_done = false;
            } else {
                cmd_pos_[idx] = grasp_target_[idx];
            }
        }
    }

    sendJointCmd();

    if (all_done) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "grasp 유지 중 | slip_events=%d",
                             slip_event_count_);
    }

    if (grasp_hold_sec_ > 0.0 &&
        (now() - grasp_start_time_).seconds() > grasp_hold_sec_) {
        RCLCPP_INFO(get_logger(), "grasp 유지 시간 초과 → SETTLING");
        transitionTo(ReflexState::SETTLING);
    }

    publishDebug();
}

// =============================================================================
// runSlipSNN
//
// [논문 핵심] FeFET-PIM 기반 SNN slip reflex
//
//   입력 전류 I = max(0, cmd_MCP - cur_MCP)
//     → cmd_MCP : 지금 보내고 있는 목표 MCP 각도
//     → cur_MCP : 실제 측정된 MCP 각도
//     → 차이가 양수 = 물체가 손가락을 밀어내고 있음 = slip
//
//   LIF 뉴런이 이 값을 시간 적분
//     → 일시적 오차는 무시 (노이즈 필터 역할)
//     → 지속적 slip 이면 막전위 누적 → threshold 초과 → spike
//
//   spike 발생 시:
//     ① grasp_target[MCP/PIP/DIP] += slip_target_increment
//        (더 깊은 목표각 → PD 위치 오차 증가 → PD 토크 자연히 증가)
//     ② fin_contacted 리셋
//        (홀드 해제 → 새 목표각으로 보간 닫기 재시작)
//
//   결과: 물체 무게/마찰/크기에 관계없이 slip 이 멈출 때까지 자동 적응
// =============================================================================
void ReflexGraspNode::runSlipSNN()
{
    for (int f = 0; f < N_FIN; f++) {
        int    mcp  = f * 4 + 1;

        // slip 입력: 목표보다 실제가 열려있는 정도
        double slip = std::max(0.0, cmd_pos_[mcp] - cur_pos_[mcp]);

        if (slip_neurons_[f].update(slip, DT)) {
            slip_event_count_++;

            RCLCPP_WARN(get_logger(),
                        "★ Slip SNN spike! finger%d  slip=%.4f rad  "
                        "→ target +%.3f rad  (총 events=%d)",
                        f, slip, slip_target_increment_, slip_event_count_);

            // ① MCP, PIP, DIP 목표각 증가 (spread 는 유지)
            for (int j = 1; j < 4; j++) {
                int idx = f * 4 + j;
                grasp_target_[idx] = std::min(
                    grasp_target_[idx] + slip_target_increment_,
                    slip_target_max_
                );
            }

            // ② 홀드 해제 → 새 목표각으로 보간 닫기 재시작
            fin_contacted_[f] = false;

            RCLCPP_INFO(get_logger(),
                        "  finger%d new target → MCP=%.3f  PIP=%.3f  DIP=%.3f",
                        f,
                        grasp_target_[f*4+1],
                        grasp_target_[f*4+2],
                        grasp_target_[f*4+3]);
        }
    }
}

// =============================================================================
// detectObjectContact
//
// cmd 는 계속 증가하는데 실제 관절 속도 급감
// → 물체 저항 → 해당 손가락 MCP 홀드
// =============================================================================
void ReflexGraspNode::detectObjectContact()
{
    for (int f = 0; f < N_FIN; f++) {
        if (fin_contacted_[f]) continue;

        int    mcp       = f * 4 + 1;
        double remaining = std::abs(grasp_target_[mcp] - cmd_pos_[mcp]);
        bool   cmd_moving = remaining > 0.01;

        double actual_vel = std::abs(cur_pos_[mcp] - prev_pos_[mcp]) / DT;
        bool   hand_stuck = actual_vel < object_contact_vel_threshold_;

        if (cmd_moving && hand_stuck) {
            fin_contacted_[f] = true;
            fin_hold_mcp_[f]  = cur_pos_[mcp];
            RCLCPP_INFO(get_logger(),
                        "★ 물체 접촉: finger%d  MCP=%.3f rad 홀드",
                        f, fin_hold_mcp_[f]);
        }
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
        fin_contacted_.fill(false);
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        initGraspTarget();   // slip 으로 증가된 target 초기화
        RCLCPP_INFO(get_logger(), "ready 발행 → %.1fs 안정화 대기",
                    settle_frames_ / 100.0);
        break;

    case ReflexState::READY_MONITORING:
        ready_pos_ = cur_pos_;   // 현재각을 기준각으로 저장
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        RCLCPP_INFO(get_logger(), "★ 기준각 저장 완료                       ★");
        RCLCPP_INFO(get_logger(), "★ 물체를 손가락에 살짝 갖다 대세요       ★");
        RCLCPP_INFO(get_logger(), "  Δθ > %.4f rad 감지 시 grasp 시작",
                    contact_delta_threshold_);
        RCLCPP_INFO(get_logger(), "────────────────────────────────────────");
        break;

    case ReflexState::SLOW_GRASPING:
        sendLibCmd("pdControl");
        cmd_pos_ = cur_pos_;     // 현재 위치에서 부드럽게 시작
        fin_contacted_.fill(false);
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        grasp_start_time_ = now();
        RCLCPP_INFO(get_logger(), "★★★ grasp 시작! ★★★");
        RCLCPP_INFO(get_logger(), "  %.2f rad/s 로 천천히 닫는 중...", grasp_speed_);
        RCLCPP_INFO(get_logger(), "  물체 잡으면 홀드 → slip 시 SNN 발화 → 더 세게");
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
    for (int i = 0; i < DOF; i++) {
        if (std::abs(cur_pos_[i] - prev_pos_[i]) > 0.003) return false;
    }
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
//  [0~3]  손가락별 Σ|Δθ|          (접촉 감지 입력값, threshold=0.05)
//  [4~7]  slip 뉴런 막전위         (이 값이 threshold 넘으면 spike)
//  [8~11] 손가락별 현재 grasp_target MCP  (slip 으로 증가하는 값)
//  [12]   slip_event_count_
//  [13]   현재 상태 (0=IDLE 1=SETTLING 2=MONITORING 3=GRASPING)
void ReflexGraspNode::publishDebug()
{
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(14);

    for (int f = 0; f < N_FIN; f++) {
        // Σ|Δθ| (접촉 감지 입력)
        double delta = 0.0;
        for (int j = 0; j < 4; j++)
            delta += std::abs(cur_pos_[f*4+j] - ready_pos_[f*4+j]);
        msg.data[f]   = delta;

        // slip 뉴런 막전위
        msg.data[f+4] = slip_neurons_[f].membrane;

        // 현재 grasp_target MCP (slip 으로 증가 추적)
        msg.data[f+8] = grasp_target_[f*4+1];
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
        case ReflexState::SLOW_GRASPING:    return "SLOW_GRASPING";
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
