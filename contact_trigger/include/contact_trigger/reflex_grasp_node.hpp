#pragma once
// =============================================================================
// reflex_grasp_node.hpp
//
// ── 핵심 설계 ─────────────────────────────────────────────────────────────────
//  allegro_node_grasp 의 setJointCallback 이
//  joint_cmd 수신 시 자동으로 eMotionType_JOINT_PD 로 전환함
//  → pdControl 명령 따로 안 보내도 됨
//  → joint_cmd 만 계속 발행하면 천천히 닫기 동작
//
// ── 상태 머신 ─────────────────────────────────────────────────────────────────
//  CALIBRATING     : grasp_3 실행 → 완전히 닫힌 관절각 자동 학습
//  SETTLING        : ready 복귀 + 안정화
//  READY_MONITORING: LIF SNN 접촉 감지 대기
//  GRASPING        : joint_cmd 보간 (천천히 닫기)
//                    slip SNN spike → grasp_target_ 굽힘 관절 += slip_push_
//
// ── 토픽 ─────────────────────────────────────────────────────────────────────
//  구독: /allegroHand_0/joint_states   현재 관절각
//  구독: /allegroHand_0/lib_cmd        키보드 'r' 감지
//  발행: /allegroHand_0/lib_cmd        "grasp_3" / "ready"
//  발행: /allegroHand_0/joint_cmd      목표 관절각
//  발행: /contact_trigger/debug        SNN 디버그
//  구독: /contact_trigger/cmd          수동 명령
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

struct LIFNeuron
{
    double membrane = 0.0, threshold = 0.012, tau = 0.05;
    bool   spiked   = false;

    bool update(double I, double dt) {
        membrane = membrane * (1.0 - dt / tau) + I * dt;
        membrane = std::max(0.0, membrane);
        if (membrane >= threshold) { spiked = true; membrane = 0.0; }
        else spiked = false;
        return spiked;
    }
    void reset() { membrane = 0.0; spiked = false; }
};

enum class ReflexState {
    IDLE,
    CALIBRATING,
    SETTLING,
    READY_MONITORING,
    GRASPING
};

class ReflexGraspNode : public rclcpp::Node
{
public:
    explicit ReflexGraspNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ReflexGraspNode() = default;

private:
    double contact_threshold_, snn_tau_contact_;
    double slip_threshold_,    snn_tau_slip_;
    int    min_finger_triggers_, settle_frames_;
    double grasp_hold_sec_, grasp_speed_, slip_push_;

    ReflexState state_;
    int  frame_count_;
    bool joint_state_received_, ignore_next_ready_;
    int  calib_stable_count_;
    bool grasp_target_learned_;

    static constexpr int    DOF = 16;
    static constexpr int  N_FIN = 4;
    static constexpr double  DT = 0.01;

    std::array<double, DOF> ready_position_;
    std::array<double, DOF> current_position_;
    std::array<double, DOF> prev_position_;
    std::array<double, DOF> cmd_position_;   // 보간 중간값
    std::array<double, DOF> grasp_target_;   // 학습된 최종 목표각

    rclcpp::Time grasp_start_time_;

    std::array<LIFNeuron, N_FIN> contact_neurons_;
    std::array<LIFNeuron, N_FIN> slip_neurons_;
    int slip_event_count_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         allegro_lib_cmd_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr            lib_cmd_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr     joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_pub_;
    rclcpp::TimerBase::SharedPtr                                   timer_;

    void jointStateCallback(const sensor_msgs::msg::JointState & msg);
    void externalCmdCallback(const std_msgs::msg::String & msg);
    void allegroLibCmdCallback(const std_msgs::msg::String & msg);
    void timerLoop();

    void tickCalibrating();
    void tickSettling();
    void tickMonitoring();
    void tickGrasping();

    int  runContactSNN();
    bool runSlipSNN();

    void sendJointCmd();
    void transitionTo(ReflexState next);
    bool isHandStable();
    void sendCmd(const std::string & cmd);
    void publishDebug();
    std::string stateName(ReflexState s) const;
};

} // namespace contact_trigger