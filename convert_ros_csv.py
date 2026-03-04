from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from pathlib import Path
import pandas as pd

bag_path = Path("rosbag2_2026_02_21-13_05_45")   # folder containing the .db3

# Load ROS2 Humble message definitions
typestore = get_typestore(Stores.ROS2_HUMBLE)

data = []

with AnyReader([bag_path], default_typestore=typestore) as reader:

    connections = [x for x in reader.connections if x.topic == "/joint_states"]

    for connection, timestamp, rawdata in reader.messages(connections):

        msg = reader.deserialize(rawdata, connection.msgtype)

        for i, name in enumerate(msg.name):
            data.append({
                "timestamp": timestamp,
                "joint": name,
                "position": msg.position[i] if i < len(msg.position) else None,
                "velocity": msg.velocity[i] if i < len(msg.velocity) else None,
                "effort": msg.effort[i] if i < len(msg.effort) else None
            })

df = pd.DataFrame(data)
df.to_csv("joint_states.csv", index=False)

print("CSV saved successfully")