#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <marine_sensor_msgs/RadarSector.h>
#include <marine_radar_control_msgs/RadarControlValue.h>
#include <sensor_msgs/Image.h>

#include <QGuiApplication>
#include <QObject>
#include <QTimer>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QImage>

#include <deque>
#include <vector>
#include <algorithm>
#include <cmath>


static const char *VERT_SHADER = R"(#version 330 core
layout(location=0) in vec4 vertex;
out vec2 texc;
uniform mat4 matrix;
void main(){
  gl_Position = matrix * vertex;
  texc = vertex.xy;
})";

static const char *FRAG_SHADER = R"(#version 330 core
in  vec2 texc;
out vec4 fragColor;
uniform sampler2D texture_sampler;
uniform float minAngle, maxAngle;
const float M_PI = 3.14159265358979323846;
void main(){
  if (texc.x == 0.0) discard;       // 중앙 수직축 제거(하얀 점 방지)
  float r = length(texc);           // 0 ~ 1
  if (r > 1.0) discard;
  float theta = atan(-texc.x, texc.y);
  if (minAngle > 0.0 && theta < 0.0) theta += 2.0*M_PI;
  if (theta < minAngle || theta > maxAngle) discard;
  fragColor = texture(texture_sampler,
                      vec2(r,(theta-minAngle)/(maxAngle-minAngle)));
})";


struct Sector {
  QImage          image;
  float           minA, maxA;
  QOpenGLTexture *tex = nullptr;
};

class OffscreenRadarRenderer : protected QOpenGLFunctions {
public:
  OffscreenRadarRenderer() : size_(0), fbo_(nullptr) {}

  bool init() {
    QSurfaceFormat fmt; fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat::CoreProfile);
    surface_.setFormat(fmt); surface_.create();
    if (!surface_.isValid()) return false;

    context_.setFormat(fmt);
    if (!context_.create()) return false;
    context_.makeCurrent(&surface_);
    initializeOpenGLFunctions();

    // 셰이더
    program_ = new QOpenGLShaderProgram();
    program_->addShaderFromSourceCode(QOpenGLShader::Vertex,   VERT_SHADER);
    program_->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAG_SHADER);
    program_->bindAttributeLocation("vertex", 0);
    if (!program_->link()) return false;
    program_->bind();
    QMatrix4x4 m; m.setToIdentity();
    program_->setUniformValue("matrix", m);
    program_->setUniformValue("texture_sampler", 0);

    // 화면 사각형 VBO
    static const GLfloat verts[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
    glGenBuffers(1, &quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    context_.doneCurrent();
    return true;
  }

  /* FBO를 (재)생성한다. newSize =  echo 수 × 2 */
  void ensureFBO(int newSize) {
    if (size_ == newSize) return;           // 이미 맞음
    context_.makeCurrent(&surface_);
    if (fbo_) delete fbo_;
    size_ = std::min(4096, std::max(64, newSize));
    QOpenGLFramebufferObjectFormat ff;
    ff.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    fbo_ = new QOpenGLFramebufferObject(size_, size_, ff);
    context_.doneCurrent();
  }

  /* RadarSector → QImage stack */
  void addSector(const marine_sensor_msgs::RadarSector& msg) {
    int H = msg.intensities.size();
    int W = msg.intensities.front().echoes.size();
    QImage polar(W, H, QImage::Format_Grayscale8);
    polar.fill(Qt::darkGray);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        polar.bits()[(H-1-i)*W + j] = uchar(msg.intensities[i].echoes[j] * 255);

    float a1 = msg.angle_start;
    float a2 = a1 + msg.angle_increment * (H - 1);
    float half = (a2 - a1) / (2.0f * H);
    float minA = a2 - half * 1.1f;      // rqt 보정
    float maxA = a1 + half * 1.1f;

    sectors_.push_back({ polar, minA, maxA, nullptr });
  }

  /* 360° 완료 → Cartesian 그림 */
  QImage renderFullCircle() {
    if (!fbo_) return QImage();          // FBO 미생성
    context_.makeCurrent(&surface_);
    fbo_->bind();
    glViewport(0, 0, size_, size_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program_->bind();
    for (auto& s : sectors_) {
      if (!s.tex) {
        s.tex = new QOpenGLTexture(s.image);
        s.tex->setMinificationFilter(QOpenGLTexture::Nearest);
        s.tex->setMagnificationFilter(QOpenGLTexture::Nearest);
        s.tex->setWrapMode(QOpenGLTexture::ClampToEdge);
      }
      program_->setUniformValue("minAngle", s.minA);
      program_->setUniformValue("maxAngle", s.maxA);

      glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
      program_->enableAttributeArray(0);
      program_->setAttributeBuffer(0, GL_FLOAT, 0, 3, 3*sizeof(GLfloat));
      s.tex->bind(0);
      glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    fbo_->release();
    QImage out = fbo_->toImage();
    context_.doneCurrent();
    return out;
  }

  void clear() {
    context_.makeCurrent(&surface_);
    for (auto& s : sectors_)
      if (s.tex) delete s.tex;
    sectors_.clear();
    context_.doneCurrent();
  }

private:
  int                               size_;
  QOffscreenSurface                 surface_;
  QOpenGLContext                    context_;
  QOpenGLFramebufferObject*         fbo_;
  QOpenGLShaderProgram*             program_;
  GLuint                            quadVBO_;
  std::deque<Sector>                sectors_;
};

/**************************************************************
* 3. Radar 상태 송신 유틸
**************************************************************/
void publishRadarKeyValue(ros::Publisher& pub,
                          const std::string& key,
                          const std::string& value,
                          int burst = 5)
{
  marine_radar_control_msgs::RadarControlValue msg;
  msg.key = key; msg.value = value;
  ros::Rate r(10);
  for (int i=0;i<burst && ros::ok(); ++i) { pub.publish(msg); r.sleep(); }
}

void sendRadarStatusSync(ros::NodeHandle& nh,
                         const std::string& topic,
                         const std::string& status_value)
{
  ros::Publisher p = nh.advertise<marine_radar_control_msgs::RadarControlValue>(topic,1,true);
  ros::Rate r(10); int tries=0;
  while (p.getNumSubscribers()==0 && ros::ok() && tries++<50) r.sleep();
  if (p.getNumSubscribers()>0) publishRadarKeyValue(p,"status",status_value);
}

/**************************************************************
* 4.   main
**************************************************************/
int main(int argc, char** argv)
{
  // Qt 오프스크린
  qputenv("QT_QPA_PLATFORM","offscreen");
  qputenv("QT_OPENGL",     "software");

  ros::init(argc, argv, "radar_img_node");
  QGuiApplication app(argc, argv);
  ros::NodeHandle nh("~");

  /* ----- 파라미터 ----- */
  std::string ctrl_topic = "/HaloA/change_state";

  std::string topic; nh.param("input_topic", topic, std::string("/HaloA/data"));
  bool   auto_cycle; nh.param("auto_cycle",  auto_cycle, false);
  double cycle_sec;  nh.param("cycle_sec",   cycle_sec,  10.0);

  std::vector<float> cycle_values;
  if (auto_cycle) {
    XmlRpc::XmlRpcValue xml_vals;
    if (nh.getParam("cycle_values", xml_vals) &&
        xml_vals.getType()==XmlRpc::XmlRpcValue::TypeArray)
      for (int i=0;i<xml_vals.size(); ++i)
        cycle_values.push_back(static_cast<double>(xml_vals[i]));
    if (cycle_values.empty()) auto_cycle = false;
  }

  /* ----- 레이더 transmit ----- */
  sendRadarStatusSync(nh, ctrl_topic, "transmit");

  /* ----- PUB/SUB, 렌더러 초기화 ----- */
  ros::Publisher img_pub = nh.advertise<sensor_msgs::Image>("radar_image",1);
  ros::Publisher range_pub =
      nh.advertise<marine_radar_control_msgs::RadarControlValue>(ctrl_topic,1,true);

  OffscreenRadarRenderer renderer;
  if (!renderer.init()) { ROS_ERROR("OpenGL init failed"); return -1; }

  /* ----- 섹터 수신 ----- */
  float last = 0.0f; bool first = true;
  const float WRAP = static_cast<float>(M_PI);
  bool fbo_ready = false;

  ros::Subscriber sector_sub =
    nh.subscribe<marine_sensor_msgs::RadarSector>(topic, 10,
      [&](const marine_sensor_msgs::RadarSector::ConstPtr& msg){
        if (!fbo_ready) {
          int W = msg->intensities.front().echoes.size();
          renderer.ensureFBO(W * 2);      // 반지름 = W
          ROS_INFO("FBO created: %d×%d (echo=%d)", W*2, W*2, W);
          fbo_ready = true;
        }
        renderer.addSector(*msg);

        float delta = msg->angle_start - last;
        if (!first && std::fabs(delta) > WRAP) {      // 360°
          QImage img = renderer.renderFullCircle();
          cv::Mat rgba(img.height(), img.width(),
                       CV_8UC4,
                       const_cast<uchar*>(img.bits()),
                       img.bytesPerLine());
          cv::Mat bgr; cv::cvtColor(rgba,bgr,cv::COLOR_RGBA2BGR);

          std_msgs::Header h; h.stamp = ros::Time::now(); h.frame_id="map";
          img_pub.publish(cv_bridge::CvImage(h,"bgr8",bgr).toImageMsg());
          renderer.clear();
        }
        first = false; last = msg->angle_start;
      });

  /* ----- 자동 range 순환(하드웨어에만 명령) ----- */
  ros::Timer cycle_timer;
  if (auto_cycle) {
    cycle_timer = nh.createTimer(ros::Duration(cycle_sec),
      [&](const ros::TimerEvent&){
        static size_t idx = 0;
        publishRadarKeyValue(range_pub, "range",
                             std::to_string(cycle_values[idx]));
        idx = (idx + 1) % cycle_values.size();
      });
  }

  /* ----- Qt 타이머로 ros::spinOnce ----- */
  QTimer qt_timer;
  QObject::connect(&qt_timer,&QTimer::timeout,[&](){
    if (!ros::ok()) {
      sendRadarStatusSync(nh, ctrl_topic, "standby");
      app.quit();
    } else ros::spinOnce();
  });
  qt_timer.start(10);   // 100 Hz

  return app.exec();
}
