#include <ros/ros.h>
#include <marine_sensor_msgs/RadarSector.h>
#include <marine_radar_control_msgs/RadarControlValue.h>
#include <sensor_msgs/CompressedImage.h>

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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <deque>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

// ------------------ GLSL ------------------
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
  float r = length(texc);
  if (r > 1.0) discard;
  float theta = atan(-texc.x, texc.y);
  if (minAngle > 0.0 && theta < 0.0) theta += 2.0*M_PI;
  bool inside;
  if (minAngle <= maxAngle) {
    inside = (theta >= minAngle && theta < maxAngle + 1e-6);
  } else {
    inside = (theta >= minAngle || theta < maxAngle + 1e-6);
  }
  if (!inside) discard;
  fragColor = texture(texture_sampler, vec2(r,(theta-minAngle)/(maxAngle-minAngle)));
})";

// ------------------ 데이터 구조 ------------------
struct Sector {
  QImage          image;
  float           minA, maxA;
  QOpenGLTexture *tex = nullptr;
};

// ------------------ 렌더러 ------------------
class OffscreenRadarRenderer : protected QOpenGLFunctions {
public:
  OffscreenRadarRenderer() : size_(0), fbo_(nullptr), program_(nullptr), quadVBO_(0) {}

  bool init() {
    QSurfaceFormat fmt; fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat::CoreProfile);
    surface_.setFormat(fmt); surface_.create();
    if (!surface_.isValid()) return false;

    context_.setFormat(fmt);
    if (!context_.create()) return false;
    context_.makeCurrent(&surface_);
    initializeOpenGLFunctions();

    program_ = new QOpenGLShaderProgram();
    program_->addShaderFromSourceCode(QOpenGLShader::Vertex,   VERT_SHADER);
    program_->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAG_SHADER);
    program_->bindAttributeLocation("vertex", 0);
    if (!program_->link()) return false;
    program_->bind();
    QMatrix4x4 m; m.setToIdentity();
    program_->setUniformValue("matrix", m);
    program_->setUniformValue("texture_sampler", 0);

    static const GLfloat verts[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
    glGenBuffers(1, &quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    context_.doneCurrent();
    return true;
  }

  void ensureFBO(int newSize) {
    if (size_ == newSize) return;
    context_.makeCurrent(&surface_);
    delete fbo_; fbo_ = nullptr;
    size_ = std::min(2048, std::max(64, newSize));
    QOpenGLFramebufferObjectFormat ff;
    ff.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    fbo_ = new QOpenGLFramebufferObject(size_, size_, ff);
    context_.doneCurrent();
  }

  // RadarSector → QImage(Grayscale8)
  void addSector(const marine_sensor_msgs::RadarSector& msg) {
    const int H = msg.intensities.size();
    const int W = msg.intensities.front().echoes.size();

    QImage polar(W, H, QImage::Format_Grayscale8);
    // 모든 픽셀 직접 채움 (초기 fill 불필요)
    for (int i = 0; i < H; ++i) {
      const auto& row = msg.intensities[i].echoes;
      uchar* dst = polar.bits() + (H-1-i)*W;
      for (int j = 0; j < W; ++j) dst[j] = static_cast<uchar>(row[j] * 255);
    }

    // start_angle을 "엣지"로 해석 (H개의 bin 폭을 정확히 커버)
    const float a_edge0 = msg.angle_start;
    const float inc     = msg.angle_increment;
    float minA = std::min(a_edge0, a_edge0 + inc * H);
    float maxA = std::max(a_edge0, a_edge0 + inc * H);

    // 미세 경계 보정
    const float EPS = 1e-4f;
    minA -= EPS; maxA += EPS;

    sectors_.push_back({ polar, minA, maxA, nullptr });
  }

  // 누적 섹터 → 카테시안 그레이스케일 이미지
  QImage renderFullCircleGray() {
    if (!fbo_) return QImage();

    context_.makeCurrent(&surface_);
    fbo_->bind();
    glViewport(0, 0, size_, size_);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program_->bind();
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    program_->enableAttributeArray(0);
    program_->setAttributeBuffer(0, GL_FLOAT, 0, 3, 3*sizeof(GLfloat));

    for (auto& s : sectors_) {
      if (!s.tex) {
        s.tex = new QOpenGLTexture(s.image);
        s.tex->setMinificationFilter(QOpenGLTexture::Nearest);
        s.tex->setMagnificationFilter(QOpenGLTexture::Nearest);
        s.tex->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::ClampToEdge);
        s.tex->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::Repeat);
      }
      program_->setUniformValue("minAngle", s.minA);
      program_->setUniformValue("maxAngle", s.maxA);
      s.tex->bind(0);
      glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    fbo_->release();
    QImage rgba = fbo_->toImage();
    context_.doneCurrent();
    return rgba.convertToFormat(QImage::Format_Grayscale8);
  }

  void clear() {
    context_.makeCurrent(&surface_);
    for (auto& s : sectors_) { delete s.tex; s.tex = nullptr; }
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

// ------------------ Radar 상태 송신 ------------------
static void publishRadarKeyValue(ros::Publisher& pub,
                                 const std::string& key,
                                 const std::string& value,
                                 int burst = 5)
{
  marine_radar_control_msgs::RadarControlValue msg;
  msg.key = key; msg.value = value;
  ros::Rate r(10);
  for (int i=0;i<burst && ros::ok(); ++i) { pub.publish(msg); r.sleep(); }
}

static void sendRadarStatusSync(ros::NodeHandle& nh,
                                const std::string& topic,
                                const std::string& status_value)
{
  ros::Publisher p = nh.advertise<marine_radar_control_msgs::RadarControlValue>(topic,1,true);
  ros::Rate r(10); int tries=0;
  while (p.getNumSubscribers()==0 && ros::ok() && tries++<50) r.sleep();
  if (p.getNumSubscribers()>0) publishRadarKeyValue(p,"status",status_value);
}

// ------------------ main ------------------
int main(int argc, char** argv)
{
  qputenv("QT_QPA_PLATFORM","offscreen");

  ros::init(argc, argv, "radar_img_node");
  QGuiApplication app(argc, argv);
  ros::NodeHandle nh("~");

  std::string input_topic;      nh.param("input_topic", input_topic, std::string("/HaloA/data"));
  std::string ctrl_topic  = "/HaloA/change_state";
  std::string compression_format; nh.param("compression_format", compression_format, std::string("png")); // "png" or "jpeg"
  int jpeg_quality;               nh.param("jpeg_quality", jpeg_quality, 90);
  std::string frame_id;           nh.param("frame_id", frame_id, std::string("map"));

  // 레이더 전송 상태 전환
  sendRadarStatusSync(nh, ctrl_topic, "transmit");

  ros::Publisher img_pub = nh.advertise<sensor_msgs::CompressedImage>("radar_image/compressed",1);

  OffscreenRadarRenderer renderer;
  if (!renderer.init()) { ROS_ERROR("OpenGL init failed"); return -1; }

  float last = 0.0f; bool first = true;
  const float WRAP = static_cast<float>(M_PI);
  bool fbo_ready = false;

  ros::Subscriber sector_sub =
    nh.subscribe<marine_sensor_msgs::RadarSector>(input_topic, 10,
      [&](const marine_sensor_msgs::RadarSector::ConstPtr& msg){
        if (!fbo_ready) {
          int W = msg->intensities.front().echoes.size();
          renderer.ensureFBO(W * 2);
          ROS_INFO("FBO created: %d×%d (echo=%d)", W*2, W*2, W);
          fbo_ready = true;
        }

        renderer.addSector(*msg);

        // 360도 랩어라운드 검출
        float delta = msg->angle_start - last;
        if (!first && std::fabs(delta) > WRAP) {
          QImage gray_q = renderer.renderFullCircleGray();

          // QImage → cv::Mat
          cv::Mat gray(gray_q.height(), gray_q.width(), CV_8UC1,
                       const_cast<uchar*>(gray_q.bits()), gray_q.bytesPerLine());
          cv::Mat gray_contig = gray.clone();

          std_msgs::Header h; 
          h.stamp = ros::Time::now(); 
          h.frame_id = frame_id;

          std::vector<uchar> buf;
          sensor_msgs::CompressedImage cmsg;
          cmsg.header = h;

          bool ok = false;
          if (compression_format == "jpeg" || compression_format == "jpg") {
            int q = std::max(0, std::min(100, jpeg_quality));
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, q};
            ok = cv::imencode(".jpg", gray_contig, buf, params);
            cmsg.format = "jpeg";
          } else {
            ok = cv::imencode(".png", gray_contig, buf);
            cmsg.format = "png";
          }

          if (!ok) {
            ROS_ERROR("cv::imencode() failed (format=%s)", cmsg.format.c_str());
            return;
          }

          cmsg.data.assign(buf.begin(), buf.end());
          img_pub.publish(cmsg);

          renderer.clear();
        }
        first = false; last = msg->angle_start;
      });

  // ROS 이벤트 펌프를 Qt 타이머로
  QTimer qt_timer;
  QObject::connect(&qt_timer,&QTimer::timeout,[&](){
    if (!ros::ok()) {
      sendRadarStatusSync(nh, ctrl_topic, "standby");
      app.quit();
    } else {
      ros::spinOnce();
    }
  });
  qt_timer.start(10);

  return app.exec();
}
