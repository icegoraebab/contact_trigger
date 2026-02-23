// =============================================================================
// reflex_grasp_node.cpp  (contact_trigger 패키지)
//
// 무촉각 뉴로모픽 그립 안정화 노드
//
// ── 핵심 기능 ────────────────────────────────────────────────────────────────
//  ① LIF SNN 접촉 감지
//     - 손가락당 1개 LIF 뉴런
//     - 입력: 관절각 오차 합산 Σ|Δθ| (Crossbar 연산 모사)
//     - 막전위 누적 → threshold 초과 → spike → grasp 트리거
//
//  ② LIF SNN Slip 감지
//     - grasp 중 손가락이 물체에서 밀려나는 속도(역방향 drift)를 입력
//     - spike 발생 시 current_grasp_speed_ 증가 → 빠르게 보상 (R-STDP 준비)
//
//  ③ 물체 크기 적응 (Object Size Detection)
//     - 천천히 닫는 도중 손가락 속도가 급감하면 물체 접촉으로 판정
//     - 해당 손가락을 그 위치에서 홀드 → 물체 크기에 자동 적응
//
//  ④ grasp_3 동일 제어
//     - pdControl 모드 전환 후 BHand grasp_3 와 동일한 목표 관절각을
//       조금씩 보간해서 전달 → 동일한 PD 토크, 다른 속도
//
// ── 토픽 ─────────────────────────────────────────────────────────────────────
//  구독: /allegroHand_0/joint_states    현재 관절각 (300Hz)
//  구독: /allegroHand_0/lib_cmd         키보드 'r' 감지
//  발행: /allegroHand_0/lib_cmd         "pdControl" / "ready" 명령
//  발행: /allegroHand_0/joint_cmd       보간된 목표 관절각
//  발행: /contact_trigger/debug         SNN 막전위·spike 디버그
//  구독: /contact_trigger/cmd           수동 명령
// =============================================================================

#include "contact_trigger/reflex_grasp_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace contact_trigger
{

// =============================================================================
// GRASP3_TARGET: BHand eMotionType_GRASP_3 와 동일한 목표 관절각 [rad]
//
// 출처: BHand 라이브러리 내부 grasp_3 목표각 (실측 기반)
// 손가락 배치:
//   finger0(index)  : joint  0~ 3  [spread, MCP, PIP, DIP]
//   finger1(middle) : joint  4~ 7
//   finger2(ring)   : joint  8~11
//   finger3(thumb)  : joint 12~15  [rotation, MCP, PIP, DIP]
//
// 실제 손에 맞게 튜닝 필요 시 이 배열만 수정할 것
// =============================================================================
static const double GRASP3_TARGET[16] = {
    // finger0 (index)   spread   MCP     PIP     DIP
                          0.00,   0.75,   0.75,   0.75,
    // finger1 (middle)
                          0.00,   0.75,   0.75,   0.75,
    // finger2 (ring)
                          0.00,   0.75,   0.75,   0.75,
    // finger3 (thumb)   rotation MCP    PIP     DIP
                          1.05,   0.50,   0.15,   0.50
};

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
  current_grasp_speed_(0.2)
{
    // ── 파라미터 선언 ────────────────────────────────────────────────────────

    // [SNN 접촉 감지]
    // 손가락 하나의 Σ|Δθ| 를 SNN 입력 전류로 사용.
    // contact_threshold 는 LIF 뉴런의 발화 임계치
    this->declare_parameter("contact_threshold",  1.0);   // LIF threshold [무차원]
    this->declare_parameter("snn_tau_contact",    0.05);  // 접촉 SNN 시상수 [s]
    this->declare_parameter("min_finger_triggers", 1);    // 최소 발화 손가락 수

    // [SNN Slip 감지]
    // grasp 중 관절이 역방향(열리는 방향)으로 drift 하는 속도를 입력.
    // slip_threshold 는 slip SNN 뉴런 발화 임계치
    this->declare_parameter("snn_tau_slip",       0.03);  // slip SNN 시상수 [s]
    this->declare_parameter("slip_threshold",     0.8);   // slip LIF threshold [무차원]

    // [물체 크기 적응]
    // 닫는 중 손가락 속도가 이 값 이하로 떨어지면 물체 접촉으로 판정
    // (단위: rad/s, 낮을수록 민감)
    this->declare_parameter("object_contact_vel_threshold", 0.005);

    // [천천히 닫기]
    // grasp_speed: 기본 닫기 속도 [rad/s]
    // grasp_speed_max: slip 발생 시 증가하는 최대 속도
    this->declare_parameter("grasp_speed",        0.15);  // rad/s
    this->declare_parameter("grasp_speed_max",    0.50);  // rad/s

    // [안정화]
    this->declare_parameter("settle_frames",      300);   // 100Hz x 3s
    this->declare_parameter("grasp_hold_sec",     0.0);   // 0=무한

    // ── 파라미터 로드 ────────────────────────────────────────────────────────
    contact_threshold_   = this->get_parameter("contact_threshold").as_double();
    snn_tau_contact_     = this->get_parameter("snn_tau_contact").as_double();
    min_finger_triggers_ = this->get_parameter("min_finger_triggers").as_int();
    snn_tau_slip_        = this->get_parameter("snn_tau_slip").as_double();
    slip_threshold_      = this->get_parameter("slip_threshold").as_double();
    grasp_speed_         = this->get_parameter("grasp_speed").as_double();
    grasp_speed_max_     = this->get_parameter("grasp_speed_max").as_double();
    settle_frames_       = this->get_parameter("settle_frames").as_int();
    grasp_hold_sec_      = this->get_parameter("grasp_hold_sec").as_double();

    current_grasp_speed_ = grasp_speed_;

    // ── SNN 뉴런 파라미터 설정 ───────────────────────────────────────────────
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
    cmd_position_.fill(0.0);
    finger_contacted_.fill(false);
    finger_hold_pos_.fill(0.0);

    initGraspTarget();

    // ── 구독 ─────────────────────────────────────────────────────────────────
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_states", 10,
        std::bind(&ReflexGraspNode::jointStateCallback, this, std::placeholders::_1));

    // 수동 명령 (reset / grasp / off / status)
    cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/contact_trigger/cmd", 1,
        std::bind(&ReflexGraspNode::externalCmdCallback, this, std::placeholders::_1));

    // 키보드 텔레옵 'r' 감지 (self-echo 방지 포함)
    allegro_lib_cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 10,
        std::bind(&ReflexGraspNode::allegroLibCmdCallback, this, std::placeholders::_1));

    // ── 발행 ─────────────────────────────────────────────────────────────────
    lib_cmd_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/allegroHand_0/lib_cmd", 1);

    // pdControl 모드에서 목표 관절각 직접 전달
    joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "/allegroHand_0/joint_cmd", 1);

    // 디버그: [finger0~3 오차, finger0~3 접촉뉴런 막전위, slip뉴런 막전위 * 4]
    debug_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/contact_trigger/debug", 10);

    // 100Hz 메인 루프
    timer_ = this->create_wall_timer(
        10ms, std::bind(&ReflexGraspNode::timerLoop, this));

    // ── 시작 로그 ─────────────────────────────────────────────────────────────
    RCLCPP_INFO(this->get_logger(), "=============================================");
    RCLCPP_INFO(this->get_logger(), "  Neuromorphic Reflex Grasp Node            ");
    RCLCPP_INFO(this->get_logger(), "  [SNN 접촉] threshold=%.2f  tau=%.3fs",
                contact_threshold_, snn_tau_contact_);
    RCLCPP_INFO(this->get_logger(), "  [SNN Slip] threshold=%.2f  tau=%.3fs",
                slip_threshold_, snn_tau_slip_);
    RCLCPP_INFO(this->get_logger(), "  [Speed]    base=%.2f rad/s  max=%.2f rad/s",
                grasp_speed_, grasp_speed_max_);
    RCLCPP_INFO(this->get_logger(), "  [Settle]   %.1fs | min_fingers=%d",
                settle_frames_ / 100.0, min_finger_triggers_);
    RCLCPP_INFO(this->get_logger(), "=============================================");

    transitionTo(ReflexState::SETTLING);
}

// =============================================================================
// initGraspTarget
// BHand grasp_3 와 동일한 목표 관절각을 grasp_target_ 에 복사
// =============================================================================
void ReflexGraspNode::initGraspTarget()
{
    for (int i = 0; i < DOF; i++) {
        grasp_target_[i] = GRASP3_TARGET[i];
    }
}

// =============================================================================
// jointStateCallback
// =============================================================================
void ReflexGraspNode::jointStateCallback(const sensor_msgs::msg::JointState & msg)
{
    if ((int)msg.position.size() < DOF) return;

    prev_position_ = current_position_;
    for (int i = 0; i < DOF; i++) {
        current_position_[i] = msg.position[i];
    }
    joint_state_received_ = true;
}

// =============================================================================
// allegroLibCmdCallback
// 키보드 'r' → "ready" 수신 → SETTLING 리셋
// self-echo 방지: 우리가 발행한 "ready" 는 ignore_next_ready_ 플래그로 무시
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
    else if (cmd == "grasp")  transitionTo(ReflexState::SLOW_GRASPING);
    else if (cmd == "off")  { sendCmd("off"); transitionTo(ReflexState::IDLE); }
    else if (cmd == "status") RCLCPP_INFO(this->get_logger(), "상태: %s", stateName(state_).c_str());
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
        case ReflexState::SETTLING:         tickSettling();     break;
        case ReflexState::READY_MONITORING: tickMonitoring();   break;
        case ReflexState::SLOW_GRASPING:    tickSlowGrasping(); break;
        case ReflexState::IDLE:             break;
    }
}

// =============================================================================
// tickSettling
// ready 포즈 안정화 대기
// settle_frames 경과 + 손 멈춤 확인 → READY_MONITORING 전이
// =============================================================================
void ReflexGraspNode::tickSettling()
{
    frame_count_++;

    if (frame_count_ >= settle_frames_ && isHandStable()) {
        transitionTo(ReflexState::READY_MONITORING);
        return;
    }

    // 타임아웃 안전장치 (settle_frames * 3 = 9초)
    if (frame_count_ >= settle_frames_ * 3) {
        RCLCPP_WARN(this->get_logger(), "안정화 타임아웃 → 강제 READY_MONITORING");
        transitionTo(ReflexState::READY_MONITORING);
    }
}

// =============================================================================
// tickMonitoring
// SNN 접촉 뉴런 동작 → spike 시 SLOW_GRASPING 전이
// =============================================================================
void ReflexGraspNode::tickMonitoring()
{
    int fired = runContactSNN();

    publishDebug();

    if (fired >= min_finger_triggers_) {
        RCLCPP_INFO(this->get_logger(),
                    "SNN 접촉 감지: %d개 손가락 발화 → SLOW_GRASPING", fired);
        transitionTo(ReflexState::SLOW_GRASPING);
    }
}

// =============================================================================
// tickSlowGrasping
// ① SNN slip 뉴런 동작 → spike 시 속도 보상
// ② 물체 크기 적응 → 손가락별 홀드 판정
// ③ 보간: cmd_position_ → grasp_target_ 방향으로 step 이동 후 발행
// =============================================================================
void ReflexGraspNode::tickSlowGrasping()
{
    // ── ① SNN slip 감지 ───────────────────────────────────────────────────
    runSlipSNN();

    // ── ② 물체 크기 적응 ──────────────────────────────────────────────────
    detectObjectContact();

    // ── ③ 보간 닫기 ───────────────────────────────────────────────────────
    const double step = current_grasp_speed_ * DT;
    bool all_done = true;

    for (int f = 0; f < N_FIN; f++) {
        if (finger_contacted_[f]) {
            // 이 손가락은 물체에 닿았으므로 홀드 위치 유지
            for (int j = 0; j < 4; j++) {
                cmd_position_[f*4 + j] = finger_hold_pos_[f];
            }
            // (MCP 만 홀드, 나머지는 target 유지)
            for (int j = 1; j < 4; j++) {
                cmd_position_[f*4 + j] = grasp_target_[f*4 + j];
            }
            continue;
        }

        // 아직 닿지 않은 손가락 → 목표각 방향으로 step 이동
        for (int j = 0; j < 4; j++) {
            int idx  = f * 4 + j;
            double diff = grasp_target_[idx] - cmd_position_[idx];

            if (std::abs(diff) > step) {
                cmd_position_[idx] += (diff > 0.0) ? step : -step;
                all_done = false;
            } else {
                cmd_position_[idx] = grasp_target_[idx];
            }
        }
    }

    // 목표 관절각 발행
    sendJointCmd();

    // 모든 손가락 완료 → 로그 (1초에 한 번)
    if (all_done) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "grasp 완료 유지 중 (slip_events=%d, speed=%.3f)",
                             slip_event_count_, current_grasp_speed_);
    }

    // grasp_hold_sec > 0 이면 자동 복귀
    if (grasp_hold_sec_ > 0.0) {
        if ((this->now() - grasp_start_time_).seconds() > grasp_hold_sec_) {
            RCLCPP_INFO(this->get_logger(), "grasp 유지 시간 초과 → SETTLING");
            transitionTo(ReflexState::SETTLING);
        }
    }

    publishDebug();
}

// =============================================================================
// runContactSNN
// 손가락당 LIF 뉴런에 Σ|Δθ| 를 입력 전류로 주입
// spike 가 발생한 손가락 수를 반환
//
// 논문: Crossbar Σ(V·G) → 적분기 → 임계치 비교 (온칩 PIM 연산)
//   V = 관절각 편차, G = 가중치(현재 1로 균일)
// =============================================================================
int ReflexGraspNode::runContactSNN()
{
    int fired_count = 0;

    for (int f = 0; f < N_FIN; f++) {
        // 손가락 하나의 관절각 오차 합산 (입력 전류)
        double input_current = 0.0;
        for (int j = 0; j < 4; j++) {
            int idx = f * 4 + j;
            input_current += std::abs(current_position_[idx] - ready_position_[idx]);
        }

        // LIF 뉴런 업데이트
        bool spike = contact_neurons_[f].update(input_current, DT);

        if (spike) {
            RCLCPP_DEBUG(this->get_logger(),
                         "접촉 SNN: finger%d SPIKE (I=%.4f)", f, input_current);
            fired_count++;
        }
    }

    return fired_count;
}

// =============================================================================
// runSlipSNN
// grasp 중 손가락이 역방향으로 drift 하는 속도를 입력 전류로 주입
// spike 발생 시 current_grasp_speed_ 를 증가시켜 slip 보상
//
// 논문: R-STDP 적응 루프의 reflex 출력 (spike-driven torque 증가)
//   - 현재는 속도 증가로 근사
//   - 향후 FeFET weight 업데이트로 교체 예정
// =============================================================================
void ReflexGraspNode::runSlipSNN()
{
    bool any_slip = false;

    for (int f = 0; f < N_FIN; f++) {
        // 현재 관절각과 목표 관절각의 차이 (이미 닫혀야 할 만큼 열려있으면 slip)
        // 기저관절(j=1) 기준으로 slip 판단
        int mcp_idx = f * 4 + 1;
        double position_error = cmd_position_[mcp_idx] - current_position_[mcp_idx];

        // 역방향 drift: 목표보다 열려있는 정도 (양수 = slip 중)
        double slip_current = std::max(0.0, position_error);

        // LIF 뉴런 업데이트
        bool spike = slip_neurons_[f].update(slip_current, DT);

        if (spike) {
            RCLCPP_WARN(this->get_logger(),
                        "Slip SNN: finger%d SPIKE (slip=%.4f rad) → 속도 보상",
                        f, slip_current);
            any_slip = true;
            slip_event_count_++;
        }
    }

    // slip spike 발생 시 현재 닫기 속도를 20% 증가 (상한: grasp_speed_max_)
    if (any_slip) {
        current_grasp_speed_ = std::min(current_grasp_speed_ * 1.2, grasp_speed_max_);
        RCLCPP_INFO(this->get_logger(),
                    "slip 보상 → speed=%.3f rad/s (total_events=%d)",
                    current_grasp_speed_, slip_event_count_);
    } else {
        // slip 없으면 서서히 기본 속도로 복귀 (지수 감쇠)
        current_grasp_speed_ = current_grasp_speed_ * 0.99 +
                               grasp_speed_ * 0.01;
    }
}

// =============================================================================
// detectObjectContact
// 물체 크기 적응: 닫는 중 손가락 기저관절 속도가 급감하면 물체 접촉 판정
// 해당 손가락은 그 위치에서 홀드 (finger_contacted_ = true)
//
// 원리:
//   - cmd_position 이 계속 증가하는데 current_position 이 따라오지 못하면
//     → 물체 저항이 있다는 뜻 → 이 위치에서 홀드
// =============================================================================
void ReflexGraspNode::detectObjectContact()
{
    const double vel_threshold = this->get_parameter("object_contact_vel_threshold").as_double();

    for (int f = 0; f < N_FIN; f++) {
        if (finger_contacted_[f]) continue;  // 이미 홀드 중인 손가락은 건너뜀

        // 기저관절(MCP) 의 실제 이동 속도 계산
        int mcp_idx = f * 4 + 1;
        double actual_vel = std::abs(current_position_[mcp_idx] - prev_position_[mcp_idx]) / DT;

        // cmd 가 계속 증가 중인데 실제 속도가 거의 0
        double cmd_vel = std::abs(cmd_position_[mcp_idx] -
                                  (cmd_position_[mcp_idx] - current_grasp_speed_ * DT));

        bool cmd_moving  = cmd_vel > 1e-6;
        bool hand_stuck  = actual_vel < vel_threshold;

        if (cmd_moving && hand_stuck) {
            // 물체 접촉으로 판정 → 이 위치에서 홀드
            finger_contacted_[f] = true;
            finger_hold_pos_[f]  = current_position_[mcp_idx];

            RCLCPP_INFO(this->get_logger(),
                        "★ 물체 크기 적응: finger%d 홀드 (MCP=%.3f rad)",
                        f, finger_hold_pos_[f]);
        }
    }
}

// =============================================================================
// sendJointCmd
// cmd_position_ 을 JointState 메시지로 변환하여 발행
// allegro_node_pd 의 pdControl 모드에서 목표 관절각으로 사용
// =============================================================================
void ReflexGraspNode::sendJointCmd()
{
    sensor_msgs::msg::JointState js;
    js.header.stamp = this->now();
    js.position.resize(DOF);
    for (int i = 0; i < DOF; i++) {
        js.position[i] = cmd_position_[i];
    }
    joint_cmd_pub_->publish(js);
}

// =============================================================================
// isHandStable
// 모든 관절의 프레임간 변화량 < 0.003 rad → 안정 판정
// SETTLING 에서 ready 포즈 도달 여부 확인에 사용
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
// /contact_trigger/debug 토픽:
//   [0~3]  손가락별 접촉 뉴런 막전위
//   [4~7]  손가락별 slip 뉴런 막전위
//   [8~11] 손가락별 관절각 오차 Σ|Δθ|
//   [12]   현재 grasp_speed_
//   [13]   slip_event_count_
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
    msg.data[12] = current_grasp_speed_;
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
        // self-echo 방지: sendCmd("ready") 직전 플래그 설정
        ignore_next_ready_ = true;
        sendCmd("ready");
        frame_count_ = 0;

        // SNN 뉴런 및 slip 카운터 리셋
        for (int f = 0; f < N_FIN; f++) {
            contact_neurons_[f].reset();
            slip_neurons_[f].reset();
        }
        slip_event_count_    = 0;
        current_grasp_speed_ = grasp_speed_;
        finger_contacted_.fill(false);

        RCLCPP_INFO(this->get_logger(),
                    "ready 명령 발행 → %.1fs + 안정화 대기",
                    settle_frames_ / 100.0);
        break;

    case ReflexState::READY_MONITORING:
        // 현재 관절각을 기준각으로 저장 (뉴런 정지전위 설정)
        ready_position_ = current_position_;

        // 접촉 SNN 뉴런 리셋
        for (int f = 0; f < N_FIN; f++) contact_neurons_[f].reset();

        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        RCLCPP_INFO(this->get_logger(), "★ 기준각 저장 완료 (뉴런 정지전위 설정) ★");
        RCLCPP_INFO(this->get_logger(), "★ SNN 접촉 감지 대기 중...             ★");
        RCLCPP_INFO(this->get_logger(), "  threshold=%.2f | tau=%.3fs | fingers>=%d",
                    contact_threshold_, snn_tau_contact_, min_finger_triggers_);
        RCLCPP_INFO(this->get_logger(), "─────────────────────────────────────");
        break;

    case ReflexState::SLOW_GRASPING:
        // pdControl 모드 전환
        // → 이후 /allegroHand_0/joint_cmd 로 목표 관절각 보내면 PD 제어 추종
        sendCmd("pdControl");

        // cmd_position 을 현재 위치로 초기화 (현재 위치에서 부드럽게 시작)
        cmd_position_ = current_position_;

        // slip SNN 리셋
        for (int f = 0; f < N_FIN; f++) slip_neurons_[f].reset();
        finger_contacted_.fill(false);

        grasp_start_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "★★★ SNN 발화! 뉴로모픽 grasp 시작 ★★★");
        RCLCPP_INFO(this->get_logger(), "  speed=%.2f rad/s (BHand grasp_3 동일 목표각)");
        RCLCPP_INFO(this->get_logger(), "  slip SNN 활성화 | 물체 크기 적응 활성화");
        RCLCPP_INFO(this->get_logger(), "  리셋: 키보드 'r'");
        break;

    case ReflexState::IDLE:
        RCLCPP_INFO(this->get_logger(), "IDLE (토크 off)");
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
        case ReflexState::SLOW_GRASPING:    return "SLOW_GRASPING";
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