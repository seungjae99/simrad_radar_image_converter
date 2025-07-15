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
#include <algorithm>
#include <cmath>


static const char *VERT_SHADER =
R"(#version 330 core
layout(location=0) in vec4 vertex;
out vec2 texc;
uniform mat4  matrix;
uniform float radialScale;
void main() {
  vec4 pos = vec4(vertex.xy * radialScale, vertex.zw);
  gl_Position = matrix * pos;
  texc = vertex.xy;
})";

static const char *FRAG_SHADER =
R"(#version 330 core
in  vec2  texc;
out vec4  fragColor;

uniform sampler2D texture_sampler;
uniform float minAngle, maxAngle;         // 섹터 각 범위
uniform float sensorRangeMax;
uniform float rangeLimit;                 // 시각화 최대거리
const float M_PI = 3.14159265358979323846;

void main() {
  float r0 = length(texc);                // 정규화 반경
  if (r0 > 1.0) discard;                  // 뷰포트 밖

  float distance = r0 * sensorRangeMax;   // 실제 거리
  if (distance > rangeLimit) discard;     // crop

  float r = distance / sensorRangeMax;    // 텍스처 r축 좌표 0 ~ 1

  float theta = atan(-texc.x, texc.y);    // [-π, π]
  if (minAngle > 0.0 && theta < 0.0) theta += 2.0 * M_PI;
  if (theta < minAngle || theta > maxAngle) discard;

  vec2 uv = vec2(r, (theta - minAngle) / (maxAngle - minAngle));
  fragColor = texture(texture_sampler, uv);
})";


struct Sector {
  QImage          image;
  float           minA, maxA;   // [rad]
  QOpenGLTexture *tex = nullptr;
};

class OffscreenRadarRenderer : protected QOpenGLFunctions {
public:
  OffscreenRadarRenderer(int fboSize, float sensorRangeMax)
    : size_(fboSize),
      sensorRangeMax_(sensorRangeMax),
      rangeLimit_(sensorRangeMax) {}

  void setSensorRangeMax(float v) { sensorRangeMax_ = v; }
  void setRangeLimit     (float v) { rangeLimit_     = v; }

  // OpenGL 초기화
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

    // FBO
    QOpenGLFramebufferObjectFormat ff;
    ff.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    fbo_ = new QOpenGLFramebufferObject(size_, size_, ff);

    context_.doneCurrent();
    return true;
  }

  // RadarSector → QImage 저장
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
    float minA = a2 - half * 1.1f;
    float maxA = a1 + half * 1.1f;

    sectors_.push_back({ polar, minA, maxA, nullptr });
  }

  // 섹터 누적 -> 하나의 Cartesian 이미지로 변환
  QImage renderFullCircle() {
    context_.makeCurrent(&surface_);
    fbo_->bind();
    glViewport(0, 0, size_, size_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program_->bind();
    program_->setUniformValue("sensorRangeMax", sensorRangeMax_);
    program_->setUniformValue("rangeLimit",     rangeLimit_);
    program_->setUniformValue("radialScale",    sensorRangeMax_ / rangeLimit_);

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
  int   size_;
  float sensorRangeMax_;   // 최대 사거리 (예: 3000 m)
  float rangeLimit_;       // 시각화 범위 (예: 500 m)
  QOffscreenSurface           surface_;
  QOpenGLContext              context_;
  QOpenGLFramebufferObject*   fbo_;
  QOpenGLShaderProgram*       program_;
  GLuint                      quadVBO_;
  std::deque<Sector>          sectors_;
};

// Radar 송수신 제어
void sendRadarStatus(ros::NodeHandle& nh,
                    const std::string& topic,
                    const std::string& status_value)
{
  ros::Publisher p = nh.advertise<marine_radar_control_msgs::RadarControlValue>(topic, 1, true);
  marine_radar_control_msgs::RadarControlValue msg;
  msg.key   = "status";
  msg.value = status_value;

  ros::Rate r(10);
  int tries = 0;
  while (p.getNumSubscribers()==0 && ros::ok() && tries++<50) { r.sleep(); }
  if (p.getNumSubscribers()>0)
    for (int i=0;i<5;++i) { p.publish(msg); r.sleep(); }
}

int main(int argc, char** argv)
{
  qputenv("QT_QPA_PLATFORM","offscreen");
  qputenv("QT_OPENGL",     "software");

  ros::init(argc, argv,"radar_img_node");
  QGuiApplication app(argc, argv);
  ros::NodeHandle nh("~");

  // 파라미터
  std::string ctrl_topic = "/HaloA/change_state";
  sendRadarStatus(nh, ctrl_topic, "transmit");

  std::string topic;            nh.param("input_topic",     topic,            std::string("/HaloA/data"));
  float       sensor_range_max; nh.param("sensor_range_max",sensor_range_max, 3000.0f);
  float       rangeLimit;       nh.param("range_limit",      rangeLimit,       1000.0f);
  double      px_per_m;         nh.param("pixels_per_meter", px_per_m,         0.08535);

  // FBO 해상도 : sensor_range_max 기준 고정
  int imageSize = std::max(64, std::min(4096,
                    static_cast<int>(std::round(sensor_range_max * px_per_m) * 2)));


  ROS_INFO("FBO imageSize=%d px (sensor_range_max=%.0fm, range_limit=%.0fm)", imageSize, sensor_range_max, rangeLimit);

  ros::Publisher pub = nh.advertise<sensor_msgs::Image>("radar_image",1);

  OffscreenRadarRenderer renderer(imageSize, sensor_range_max);
  if (!renderer.init()) { ROS_ERROR("OpenGL init failed"); return -1; }
  renderer.setRangeLimit(rangeLimit);     // 초기 crop 범위 적용

  // RadarSector 콜백
  float last=0.0f;
  bool  first=true;
  const float WRAP = static_cast<float>(M_PI);

  ros::Subscriber sub =
    nh.subscribe<marine_sensor_msgs::RadarSector>(topic, 10,
      [&](const marine_sensor_msgs::RadarSector::ConstPtr& msg){
        renderer.addSector(*msg);
        renderer.setSensorRangeMax(msg->range_max);   // 실시간 갱신

        float delta = msg->angle_start - last;
        if (!first && std::fabs(delta) > WRAP) {      // 한 바퀴 완료
          QImage img = renderer.renderFullCircle();

          cv::Mat rgba(img.height(), img.width(),
                      CV_8UC4,
                      const_cast<uchar*>(img.bits()),
                      img.bytesPerLine());
          cv::Mat bgr; cv::cvtColor(rgba,bgr,cv::COLOR_RGBA2BGR);

          std_msgs::Header h; h.stamp = ros::Time::now(); h.frame_id="map";
          pub.publish(cv_bridge::CvImage(h,"bgr8",bgr).toImageMsg());
          renderer.clear();
        }
        first = false;
        last  = msg->angle_start;
      });

  QTimer timer;
  QObject::connect(&timer,&QTimer::timeout,[&](){
    if (!ros::ok()) {
      sendRadarStatus(nh, ctrl_topic, "standby");
      app.quit();
    } else {
      ros::spinOnce();
    }
  });
  timer.start(10);

  return app.exec();
}
