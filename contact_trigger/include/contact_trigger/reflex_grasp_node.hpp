#pragma once
// =============================================================================
// reflex_grasp_node.hpp  (contact_trigger 패키지)
//
// 무촉각 뉴로모픽 그립 안정화 노드
//
// ── 논문 연결 ────────────────────────────────────────────────────────────────
//  ① 접촉 감지     : Σ|Δθ| > threshold 단순 비교 → grasp 트리거
//  ② LIF SNN Slip  : grasp 중 역방향 drift 누적 → spike → target 증가 → 토크↑
//  ③ 물체 크기 적응 : 실제 관절 속도 급감 → 해당 손가락 홀드
//  ④ 천천히 닫기   : grasp_3 목표각을 step 보간 전달
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
// LIF 뉴런  — Slip 감지 전용
//   dV/dt = -V/tau + I_in
//   V ≥ threshold → spike, V 리셋
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
    SETTLING,           // ready 포즈 안정화 대기
    READY_MONITORING,   // 접촉 감지 대기 (단순 Δθ 비교)
    SLOW_GRASPING       // 천천히 닫기 + SNN slip 감지
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
    static constexpr double DT    = 0.01;   // 100Hz

    // 파라미터
    double contact_delta_threshold_;    // 접촉 감지 Δθ 임계값
    int    min_finger_triggers_;
    double slip_threshold_;             // slip LIF 발화 임계치
    double snn_tau_slip_;               // slip SNN 시상수
    double slip_target_increment_;      // spike 시 목표각 증가량
    double slip_target_max_;            // 목표각 상한
    double grasp_speed_;
    double object_contact_vel_threshold_;
    int    settle_frames_;
    double grasp_hold_sec_;

    // 상태
    ReflexState state_             = ReflexState::IDLE;
    int         frame_count_       = 0;
    bool        js_received_       = false;
    bool        ignore_next_ready_ = false;
    int         slip_event_count_  = 0;

    // 관절각
    std::array<double, DOF> ready_pos_{};
    std::array<double, DOF> cur_pos_{};
    std::array<double, DOF> prev_pos_{};
    std::array<double, DOF> cmd_pos_{};
    std::array<double, DOF> grasp_target_{};   // slip 발생 시 동적으로 증가

    // 물체 크기 적응
    std::array<bool,   N_FIN> fin_contacted_{};
    std::array<double, N_FIN> fin_hold_mcp_{};

    rclcpp::Time grasp_start_time_;

    // SNN 뉴런 — slip 감지 전용
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
    void tickMonitoring();   // 단순 Δθ 비교
    void tickGrasping();     // SNN slip 감지

    // SNN
    void runSlipSNN();

    // 물체 크기 적응
    void detectObjectContact();

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
