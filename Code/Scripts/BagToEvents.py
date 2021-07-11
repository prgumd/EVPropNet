#!/usr/bin/python

import rosbag
import sys
import argparse
from tqdm import tqdm

def main(Args):
    print("Processing file: " + Args.BagFileName)

    i = 0
    with rosbag.Bag(Args.BagFileName, 'r') as bag:
      target = open(Args.WriteFileName, 'w')

      NumEvents = bag.get_message_count(Args.TopicName)
      for topic, msg, t in tqdm(bag.read_messages()):
        if topic == Args.TopicName:
          for e in msg.events:
            i = i + 1
            target.write(str(e.x) + "," + str(e.y) + "," + str(1 if e.polarity else 0) + "," + str(e.ts.to_nsec()/1000) + "\n")
      target.close()
      print("Wrote " + str(i) + " events into text file " + Args.BagFileName + ".txt")


if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BagFileName', default='./EVPropNet.bag', help='Path to bag file including file name and extension, Default: ./EVPropNet.bag')
    Parser.add_argument('--TopicName', default='/samsung/left/camera/events', help='ROS Topic Name where events are published, Default: /samsung/left/camera/events')
    Parser.add_argument('--WriteFileName', default='./EVPropNet.csv', \
        help='Path to save csv file including file name and extension, Default: ./EVPropNet.csv')

    Args = Parser.parse_args()
    main(Args)


