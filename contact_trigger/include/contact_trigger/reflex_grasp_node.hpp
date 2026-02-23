#pragma once
// =============================================================================
// reflex_grasp_node.hpp  (contact_trigger 패키지)
//
// ── 동작 흐름 ────────────────────────────────────────────────────────────────
//  SETTLING        : "ready" 발행 → 손 안정화
//  READY_MONITORING: Δθ 감지 → 물체 접촉 → GRASPING
//  GRASPING        : "grasp_3" 발행 → 포즈 완료 대기
//  SNN_STABILIZING : 토크 부하 감지 → SNN slip 보정
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
// LIF 뉴런 — Slip 감지 전용
// =============================================================================
struct LIFNeuron
{
    double membrane  = 0.0;
    double threshold = 1.0;
    double tau       = 0.03;
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
    SETTLING,           // ready 포즈 안정화
    READY_MONITORING,   // 접촉 감지 대기
    GRASPING,           // grasp_3 포즈 이동 중
    SNN_STABILIZING     // 토크 부하 감지 후 SNN slip 보정
};

// =============================================================================
// ReflexGraspNode
// =============================================================================
class ReflexGraspNode : public rclcpp::Node
{
public:
    explicit ReflexGraspNode(
        const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ReflexGraspNode() = default;

private:
    static constexpr int    DOF   = 16;
    static constexpr int    N_FIN = 4;
    static constexpr double DT    = 0.01;

    // 파라미터
    double contact_delta_threshold_;
    int    min_finger_triggers_;
    double grasp_reach_threshold_;
    double effort_contact_threshold_;
    int    min_effort_fingers_;
    double slip_threshold_;
    double snn_tau_slip_;
    double slip_target_increment_;
    double slip_target_max_;
    int    settle_frames_;
    double grasp_hold_sec_;

    // 상태
    ReflexState state_             = ReflexState::IDLE;
    int         frame_count_       = 0;
    bool        js_received_       = false;
    bool        ignore_next_ready_ = false;
    int         slip_event_count_  = 0;

    // 관절각 / 토크
    std::array<double, DOF> ready_pos_{};
    std::array<double, DOF> cur_pos_{};
    std::array<double, DOF> prev_pos_{};
    std::array<double, DOF> cur_effort_{};   // 현재 토크
    std::array<double, DOF> cmd_pos_{};
    std::array<double, DOF> grasp_target_{}; // slip 시 동적 증가

    rclcpp::Time grasp_start_time_;

    // SNN 뉴런 — slip 전용
    std::array<LIFNeuron, N_FIN> slip_neurons_;

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         lib_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr            lib_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr     jcmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dbg_pub_;
    rclcpp::TimerBase::SharedPtr                                   timer_;

    // 콜백
    void onJointState(const sensor_msgs::msg::JointState & msg);
    void onExtCmd    (const std_msgs::msg::String & msg);
    void onLibCmd    (const std_msgs::msg::String & msg);
    void timerCb     ();

    // 상태별 tick
    void tickSettling();
    void tickMonitoring();
    void tickGrasping();
    void tickSnnStabilizing();

    // SNN
    void runSlipSNN();

    // 판정
    bool isGraspReached() const;
    bool isEffortDetected() const;

    // 유틸
    void        transitionTo(ReflexState next);
    bool        isStable() const;
    void        initGraspTarget();
    void        sendLibCmd(const std::string & cmd);
    void        sendJointCmd();
    void        publishDebug();
    std::string stateName(ReflexState s) const;
};

}  // namespace contact_trigger
