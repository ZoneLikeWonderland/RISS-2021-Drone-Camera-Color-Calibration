/*
 * Copyright (C) 2008, Morgan Quigley and Willow Garage, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the names of Stanford University or Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// %Tag(FULLTEXT)%
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <camera_color_fe/colorir.h>
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>
#include <chrono>

using namespace InferenceEngine;

ros::Publisher retailer_pub;

std::string input_name;
std::string output_name;
ExecutableNetwork executable_network;

InferenceEngine::Blob::Ptr wrapMat2Blob_f32(const cv::Mat &mat)
{
  size_t channels = mat.channels();
  size_t height = mat.size().height;
  size_t width = mat.size().width;

  InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                                    {1, channels, height, width},
                                    InferenceEngine::Layout::NHWC);

  return InferenceEngine::make_shared_blob<float>(tDesc, (float *)mat.data);
}

sensor_msgs::Image Mat2Image(cv::Mat frame_now, int channel = 3)
{

  sensor_msgs::Image output_image_msg;

  output_image_msg.height = frame_now.rows;
  output_image_msg.width = frame_now.cols;
  output_image_msg.encoding = "bgr8";
  output_image_msg.is_bigendian = false;
  output_image_msg.step = frame_now.cols * channel;
  size_t size = output_image_msg.step * frame_now.rows;
  output_image_msg.data.resize(size);
  memcpy((char *)(&output_image_msg.data[0]), frame_now.data, size);

  return output_image_msg;
}

void imageCallback(const sensor_msgs::Image::ConstPtr &image_msg)
{
  //cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
  //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
  ROS_INFO("I heard a image %dx%d", image_msg->width, image_msg->height);
  cv::Mat bayer_mat(image_msg->height, image_msg->width, CV_8U, const_cast<uchar *>(&image_msg->data[0]), image_msg->step);
  cv::Mat color_mat;

  cv::cvtColor(bayer_mat, color_mat, cv::COLOR_BayerRG2BGR);
  cv::resize(color_mat, color_mat, {color_mat.cols / 2, color_mat.rows / 2});

  InferRequest infer_request = executable_network.CreateInferRequest();
  cv::Mat image_f32;
  color_mat.convertTo(image_f32, CV_32FC3, 1.0 / 255);
  Blob::Ptr imgBlob = wrapMat2Blob_f32(image_f32); // just wrap Mat data by Blob::Ptr without allocating of new memory
  infer_request.SetBlob(input_name, imgBlob);      // infer_request accepts input blob of any size

  auto start = std::chrono::steady_clock::now();
  //            std::cout<<clock()*1.0/CLOCKS_PER_SEC<<"\n";
  infer_request.Infer();
  //            std::cout<<clock()*1.0/CLOCKS_PER_SEC<<"\n";
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout << "time = " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";

  Blob::Ptr output = infer_request.GetBlob(output_name);
  auto raw_ptr = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
  auto outbox = cv::Mat(480, 640, CV_32F, raw_ptr);
  // cv::imwrite("outbox.png",outbox*255);
  auto outptr = cv::Mat(480, 640, CV_32F, raw_ptr + 640 * 480);
  //  cv::imwrite("outptr.png",outptr*255);

  cv::Mat outbox_8u;
  outbox.convertTo(outbox_8u, CV_8UC1, 255);

  cv::Mat outptr_8u;
  outptr.convertTo(outptr_8u, CV_8UC1, 255);

  camera_color_fe::colorir ir;

  ir.raw = Mat2Image(color_mat);
  ir.block = Mat2Image(outbox_8u, 1);
  ir.point = Mat2Image(outptr_8u, 1);

  retailer_pub.publish(ir);
}
// %EndTag(CALLBACK)%

int main(int argc, char **argv)
{

  Core ie;
  std::string input_model = "/zlwtest/deploy/card/card.xml";
  CNNNetwork network = ie.ReadNetwork(input_model);

  InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
  input_name = network.getInputsInfo().begin()->first;

  input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
  input_info->setLayout(Layout::NHWC);
  input_info->setPrecision(Precision::FP32);

  DataPtr output_info = network.getOutputsInfo().begin()->second;
  output_name = network.getOutputsInfo().begin()->first;

  output_info->setPrecision(Precision::FP32);

  std::string device_name = "CPU";
  executable_network = ie.LoadNetwork(network, device_name);

  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "retailer");

  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the last
   * NodeHandle destructed will close down the node.
   */
  ros::NodeHandle n;

  /**
   * The subscribe() call is how you tell ROS that you want to receive messages
   * on a given topic.  This invokes a call to the ROS
   * master node, which keeps a registry of who is publishing and who
   * is subscribing.  Messages are passed to a callback function, here
   * called chatterCallback.  subscribe() returns a Subscriber object that you
   * must hold on to until you want to unsubscribe.  When all copies of the Subscriber
   * object go out of scope, this callback will automatically be unsubscribed from
   * this topic.
   *
   * The second parameter to the subscribe() function is the size of the message
   * queue.  If messages are arriving faster than they are being processed, this
   * is the number of messages that will be buffered up before beginning to throw
   * away the oldest ones.
   */
  // %Tag(SUBSCRIBER)%
  //retailer_pub = n.advertise<sensor_msgs::Image>("retailer", 1);
  retailer_pub = n.advertise<camera_color_fe::colorir>("retailer", 1);
  ros::Subscriber sub = n.subscribe("/xic_stereo/left/image_raw", 1, imageCallback);
  // %EndTag(SUBSCRIBER)%

  /**
   * ros::spin() will enter a loop, pumping callbacks.  With this version, all
   * callbacks will be called from within this thread (the main one).  ros::spin()
   * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
   */
  // %Tag(SPIN)%
  ros::spin();
  // %EndTag(SPIN)%

  return 0;
}
// %EndTag(FULLTEXT)%
