from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from pathlib import Path
import pandas as pd

bag_path = Path("rosbag2_2026_02_21-13_05_45")
typestore = get_typestore(Stores.ROS2_HUMBLE)

rows = []

with AnyReader([bag_path], default_typestore=typestore) as reader:
    conns = [c for c in reader.connections if c.topic == "/joint_states"]

    for conn, t, raw in reader.messages(conns):
        msg = reader.deserialize(raw, conn.msgtype)

        # ---- FILTER: skip "empty" joint states ----
        # If velocity and effort arrays are empty, it's usually the dummy publisher.
        if len(msg.velocity) == 0 and len(msg.effort) == 0:
            continue

        # Another safeguard: if all positions are exactly 0.0, skip
        if len(msg.position) > 0 and all(abs(p) < 1e-12 for p in msg.position):
            continue

        row = {"timestamp": t}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                row[f"{name}_pos"] = msg.position[i]
            if i < len(msg.velocity):
                row[f"{name}_vel"] = msg.velocity[i]
            if i < len(msg.effort):
                row[f"{name}_eff"] = msg.effort[i]

        rows.append(row)

df = pd.DataFrame(rows).sort_values("timestamp")

# optional: convert timestamp (ns) to seconds relative to start
t0 = df["timestamp"].iloc[0]
df["t_sec"] = (df["timestamp"] - t0) * 1e-9

df.to_csv("joint_states_filtered_wide.csv", index=False)
print("Saved: joint_states_filtered_wide.csv")