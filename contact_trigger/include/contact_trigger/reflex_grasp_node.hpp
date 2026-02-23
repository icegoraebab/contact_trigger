#pragma once
// =============================================================================
// reflex_grasp_node.hpp  (contact_trigger 패키지)
//
// 무촉각 뉴로모픽 그립 안정화 노드
//
// ── 논문 연결 ────────────────────────────────────────────────────────────────
//  ① LIF SNN 접촉 감지  : 관절각 오차 → 막전위 적분 → 임계 발화
//  ② LIF SNN Slip 감지  : grasp 중 관절각 역방향 drift → 발화 → 토크↑
//  ③ 물체 크기 적응      : 닫는 중 속도 저항 감지 → 그 위치에서 홀드
//  ④ 천천히 닫기 (보간)  : grasp_3 동일 목표각을 조금씩 보간 전달
// =============================================================================

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <array>
#include <cmath>
#include <string>

namespace contact_trigger
{

// =============================================================================
// LIF (Leaky Integrate-and-Fire) 뉴런 구조체
//
// 논문: FeFET crossbar 의 적분기 + 임계치 회로를 소프트웨어로 모사
//   dV/dt = -V/tau + I_in
//   V > threshold → spike 발화 → V 리셋
// =============================================================================
struct LIFNeuron
{
    double membrane   = 0.0;   // 현재 막전위 [V]
    double threshold  = 1.0;   // 발화 임계치
    double tau        = 0.05;  // 막전위 감쇠 시상수 [s]
    double leak_rate  = 0.0;   // 계산된 leak (1/tau * dt)
    bool   spiked     = false; // 이번 스텝 발화 여부

    // dt [s] 마다 호출, input_current 는 입력 신호
    bool update(double input_current, double dt)
    {
        leak_rate = dt / tau;
        membrane  = membrane * (1.0 - leak_rate) + input_current * dt;
        membrane  = std::max(0.0, membrane);   // 음수 방지

        if (membrane >= threshold) {
            spiked   = true;
            membrane = 0.0;   // 리셋
        } else {
            spiked   = false;
        }
        return spiked;
    }

    void reset() { membrane = 0.0; spiked = false; }
};

// =============================================================================
// 상태 머신
// =============================================================================
enum class ReflexState
{
    IDLE,              // 비활성 (토크 off)
    SETTLING,          // ready 포즈 안정화 대기
    READY_MONITORING,  // 접촉 감지 중  (SNN 접촉 뉴런 동작)
    SLOW_GRASPING      // 천천히 닫는 중 (SNN slip 뉴런 동작)
};

// =============================================================================
// ReflexGraspNode
// =============================================================================
class ReflexGraspNode : public rclcpp::Node
{
public:
    explicit ReflexGraspNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ReflexGraspNode() = default;

private:
    // ── 파라미터 ────────────────────────────────────────────────────────────
    double contact_threshold_;      // SNN 접촉 뉴런 발화 임계치 [rad]
    double snn_tau_contact_;        // 접촉 SNN 시상수 [s]
    double snn_tau_slip_;           // slip SNN 시상수 [s]
    double slip_threshold_;         // slip 발화 임계치 [rad/s]
    int    min_finger_triggers_;    // 최소 발화 손가락 수
    int    settle_frames_;          // 안정화 대기 프레임 (100Hz)
    double grasp_speed_;            // 기본 닫기 속도 [rad/s]
    double grasp_speed_max_;        // slip 발생 시 최대 속도 [rad/s]
    double grasp_hold_sec_;         // grasp 유지 시간 [s] (0=무한)

    // ── 상태 ────────────────────────────────────────────────────────────────
    ReflexState state_;
    int         frame_count_;
    bool        joint_state_received_;
    bool        ignore_next_ready_;   // self-echo 방지 플래그

    static constexpr int DOF    = 16;
    static constexpr int N_FIN  = 4;   // 손가락 수
    static constexpr double DT  = 0.01; // 100Hz → dt

    // ── 관절각 배열 ──────────────────────────────────────────────────────────
    std::array<double, DOF> ready_position_;    // 기준 관절각 (뉴런 정지전위)
    std::array<double, DOF> current_position_;  // 최신 관절각
    std::array<double, DOF> prev_position_;     // 이전 프레임 관절각
    std::array<double, DOF> cmd_position_;      // 현재 보내는 목표 관절각
    std::array<double, DOF> grasp_target_;      // 최종 grasp 목표각

    // ── 물체 크기 적응 ───────────────────────────────────────────────────────
    // 손가락별로 물체에 닿았으면 더 이상 닫지 않음
    std::array<bool,   N_FIN> finger_contacted_; // 손가락별 물체 접촉 여부
    std::array<double, N_FIN> finger_hold_pos_;  // 접촉 시 홀드할 관절각(기저관절)

    rclcpp::Time grasp_start_time_;

    // ── SNN 뉴런 배열 ────────────────────────────────────────────────────────
    // 접촉 감지용 (손가락당 1 뉴런)
    std::array<LIFNeuron, N_FIN> contact_neurons_;
    // slip 감지용 (손가락당 1 뉴런)
    std::array<LIFNeuron, N_FIN> slip_neurons_;

    // slip 발생 횟수 카운터 (R-STDP 확장 포인트)
    int slip_event_count_;

    // 현재 적용 중인 닫기 속도 (slip 시 증가)
    double current_grasp_speed_;

    // ── ROS 인터페이스 ───────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         allegro_lib_cmd_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr            lib_cmd_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr     joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_pub_;
    rclcpp::TimerBase::SharedPtr                                   timer_;

    // ── 콜백 ────────────────────────────────────────────────────────────────
    void jointStateCallback(const sensor_msgs::msg::JointState & msg);
    void externalCmdCallback(const std_msgs::msg::String & msg);
    void allegroLibCmdCallback(const std_msgs::msg::String & msg);
    void timerLoop();

    // ── 상태별 tick ──────────────────────────────────────────────────────────
    void tickSettling();
    void tickMonitoring();
    void tickSlowGrasping();

    // ── SNN 핵심 함수 ────────────────────────────────────────────────────────
    int  runContactSNN();             // 접촉 SNN → 발화 손가락 수 반환
    void runSlipSNN();                // slip SNN → 발화 시 속도 증가
    void detectObjectContact();       // 물체 크기 적응: 저항 감지 시 홀드

    // ── 유틸 ────────────────────────────────────────────────────────────────
    void transitionTo(ReflexState next);
    bool isHandStable();
    void initGraspTarget();
    void sendCmd(const std::string & cmd);
    void sendJointCmd();
    void publishDebug();
    std::string stateName(ReflexState s) const;
};

}  // namespace contact_trigger