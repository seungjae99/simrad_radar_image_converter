#!/usr/bin/env python3
import rospy
from marine_radar_control_msgs.msg import RadarControlValue

def main():
    rospy.init_node('radar_set_state')
    range = rospy.get_param('~range', '1500')          # 기본 standby
    topic = rospy.get_param('~topic', '/HaloA/change_state')

    pub = rospy.Publisher(topic, RadarControlValue, queue_size=1, latch=True)
    msg = RadarControlValue()
    msg.key = "range"
    msg.value = range

    # 0.5 s 동안 5 회 정도 반복 송신 후 종료
    rate = rospy.Rate(10)
    for _ in range(5):
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
