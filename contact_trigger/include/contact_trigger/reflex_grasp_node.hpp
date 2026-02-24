#pragma once
// =============================================================================
// reflex_grasp_node.hpp  (contact_trigger 패키지)
//
// 무촉각 뉴로모픽 그립 노드
//
// ── 설계 ─────────────────────────────────────────────────────────────────────
//  접촉 감지 : LIF SNN → 관절각 오차 적분 → spike → grasp_3 트리거
//  grasp 실행: allegro_node_grasp 의 "grasp_3" 명령 그대로 사용
//  토크 적응 : slip SNN spike → envelop_torque 조금 증가
//              slip 없으면 → envelop_torque 서서히 감소
//              → slip이 안 일어날 최소 토크로 자동 수렴
//
// ── 토픽 ─────────────────────────────────────────────────────────────────────
//  구독: /allegroHand_0/joint_states      현재 관절각
//  구독: /allegroHand_0/lib_cmd           키보드 'r' 감지
//  발행: /allegroHand_0/lib_cmd           "ready" / "grasp_3"
//  발행: /allegroHand_0/envelop_torque    slip 적응 토크
//  발행: /contact_trigger/debug           SNN 디버그
//  구독: /contact_trigger/cmd             수동 명령
// =============================================================================

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <array>
#include <cmath>
#include <string>

namespace contact_trigger
{

// =============================================================================
// LIF (Leaky Integrate-and-Fire) 뉴런
// 논문: FeFET crossbar 적분기 + 임계치 회로 소프트웨어 모사
//   V(t+dt) = V(t) × (1 - dt/tau) + I × dt
//   V ≥ threshold → spike → V 리셋
// =============================================================================
struct LIFNeuron
{
    double membrane  = 0.0;
    double threshold = 0.012;
    double tau       = 0.05;
    bool   spiked    = false;

    bool update(double input_current, double dt)
    {
        membrane = membrane * (1.0 - dt / tau) + input_current * dt;
        membrane = std::max(0.0, membrane);
        if (membrane >= threshold) {
            spiked   = true;
            membrane = 0.0;
        } else {
            spiked = false;
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
    IDLE,
    SETTLING,
    READY_MONITORING,
    GRASPING
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
    double contact_threshold_;
    double snn_tau_contact_;
    double snn_tau_slip_;
    double slip_threshold_;
    int    min_finger_triggers_;
    int    settle_frames_;
    double grasp_hold_sec_;

    // ── envelop torque 적응 제어 ─────────────────────────────────────────────
    // slip 없으면 서서히 감소, slip 발생 시 조금 증가
    // → slip이 안 일어날 최소 토크로 자동 수렴
    double envelop_torque_;     // 현재 적용 중인 torque  [0.0 ~ 1.0]
    double torque_min_;         // 최소 토크 한계
    double torque_max_;         // 최대 토크 한계 (과도한 힘 방지)
    double torque_step_up_;     // slip spike 1회당 증가량
    double torque_decay_rate_;  // slip 없을 때 지수 감쇠 비율 (per tick)

    // ── 상태 ────────────────────────────────────────────────────────────────
    ReflexState state_;
    int         frame_count_;
    bool        joint_state_received_;
    bool        ignore_next_ready_;

    static constexpr int    DOF   = 16;
    static constexpr int    N_FIN = 4;
    static constexpr double DT    = 0.01;

    std::array<double, DOF> ready_position_;
    std::array<double, DOF> current_position_;
    std::array<double, DOF> prev_position_;

    rclcpp::Time grasp_start_time_;

    // ── SNN 뉴런 ─────────────────────────────────────────────────────────────
    std::array<LIFNeuron, N_FIN> contact_neurons_;
    std::array<LIFNeuron, N_FIN> slip_neurons_;
    int slip_event_count_;

    // ── ROS 인터페이스 ───────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         allegro_lib_cmd_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr            lib_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr           envelop_torque_pub_;
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
    void tickGrasping();

    // ── SNN ──────────────────────────────────────────────────────────────────
    int  runContactSNN();
    bool runSlipSNN();          // slip 발생 여부 반환

    // ── 토크 적응 ────────────────────────────────────────────────────────────
    void updateEnvelopTorque(bool slip_occurred);
    void publishEnvelopTorque();
    void resetEnvelopTorque();

    // ── 유틸 ─────────────────────────────────────────────────────────────────
    void transitionTo(ReflexState next);
    bool isHandStable();
    void sendCmd(const std::string & cmd);
    void publishDebug();
    std::string stateName(ReflexState s) const;
};

}  // namespace contact_trigger