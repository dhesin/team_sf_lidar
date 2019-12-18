pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null
trap "exit" INT TERM
trap "kill 0" INT
rosparam set use_sim_time true
roslaunch $SCRIPTPATH/../../didi_competition/mkz-description/tf.launch & 
pid[0]=$!
roslaunch velodyne_pointcloud 32e_points.launch &
pid[1]=$!
/opt/ros/indigo/lib/rosbag/record -a -O $2 &
pid[2]=$!
#trap " kill -SIGINT ${pid[2]} ${pid[1]} ${pid[0]}; exit 1" INT
/opt/ros/indigo/lib/rosbag/play --delay=5 --queue=0 --rate=0.05 --clock --quiet $1
sleep 1

kill -SIGINT ${pid[2]} ${pid[1]} ${pid[0]}
wait
if [ -f $2.active ]; then
   rm $2.active
fi
