```
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right
```

```
ros2 run allegro_hand_keyboards allegro_hand_keyboard --ros-args -r /allegroHand/lib_cmd:=/allegroHand_0/lib_cmd
```

```
ros2 launch contact_trigger reflex_grasp.launch.py
```
