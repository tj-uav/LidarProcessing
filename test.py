from pathlib import Path
from rosbags.highlevel import AnyReader

bagpath = Path('HolybroOut02.bag')

with AnyReader([bagpath]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        msg = reader.deserialize(rawdata, connection.msgtype)
        print(msg)