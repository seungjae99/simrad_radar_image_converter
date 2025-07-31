#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
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
  if (texc.x == 0.0) discard;       // 중앙 수직축 제거(중앙 점 방지)
  float r = length(texc);           // 0 ~ 1
  if (r > 1.0) discard;
  float theta = atan(-texc.x, texc.y);
  if (minAngle > 0.0 && theta < 0.0) theta += 2.0*M_PI;
  if (theta < minAngle || theta > maxAngle) discard;
  // 입력 텍스처는 그레이스케일이지만 GL에서는 RGBA로 샘플 → 단일 채널만 사용되어도 됨.
  vec4 s = texture(texture_sampler, vec2(r,(theta-minAngle)/(maxAngle-minAngle)));
  fragColor = vec4(s.rrr, 1.0);   // grayscale → RGB 동일값
})";

// ------------------ 데이터 구조 ------------------
struct Sector {
  QImage          image;   // Format_Grayscale8
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
    if (!surface_.isValid()) { ROS_ERROR("Invalid offscreen surface"); return false; }

    context_.setFormat(fmt);
    if (!context_.create()) { ROS_ERROR("OpenGL context create failed"); return false; }
    context_.makeCurrent(&surface_);
    initializeOpenGLFunctions();

    // Shader
    program_ = new QOpenGLShaderProgram();
    program_->addShaderFromSourceCode(QOpenGLShader::Vertex,   VERT_SHADER);
    program_->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAG_SHADER);
    program_->bindAttributeLocation("vertex", 0);
    if (!program_->link()) { ROS_ERROR("Shader link failed"); return false; }
    program_->bind();
    QMatrix4x4 m; m.setToIdentity();
    program_->setUniformValue("matrix", m);
    program_->setUniformValue("texture_sampler", 0);

    // Fullscreen quad VBO (x,y,z)
    static const GLfloat verts[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
    glGenBuffers(1, &quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    context_.doneCurrent();
    return true;
  }

  void ensureFBO(int newSize) {
    if (size_ == newSize && fbo_) return;
    context_.makeCurrent(&surface_);
    if (fbo_) { delete fbo_; fbo_ = nullptr; }
    size_ = std::min(4096, std::max(64, newSize));
    QOpenGLFramebufferObjectFormat ff;
    ff.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    fbo_ = new QOpenGLFramebufferObject(size_, size_, ff);
    glViewport(0, 0, size_, size_);
    context_.doneCurrent();
  }

  // RadarSector → QImage(Grayscale8)
  void addSector(const marine_sensor_msgs::RadarSector& msg) {
    int H = msg.intensities.size();
    int W = msg.intensities.front().echoes.size();

    QImage polar(W, H, QImage::Format_Grayscale8);
    uchar* ptr = polar.bits();
    for (int i = 0; i < H; ++i) {
      const auto& row = msg.intensities[i].echoes;
      for (int j = 0; j < W; ++j) {
        // 메시지 스케일이 [0,1] 가정 → 0~255로 매핑
        ptr[(H-1-i)*W + j] = static_cast<uchar>(std::max(0.0f, std::min(1.0f, row[j])) * 255.0f);
      }
    }

    // 각도(라디안)
    float a1 = msg.angle_start;
    float a2 = a1 + msg.angle_increment * (H - 1);
    float half = (a2 - a1) / (2.0f * H);
    float minA = a2 - half * 1.1f;  // 보정
    float maxA = a1 + half * 1.1f;

    sectors_.push_back({ polar, minA, maxA, nullptr });
  }

  // 누적 섹터 → 카테시안 그레이스케일 이미지(QImage::Format_Grayscale8)
  QImage renderFullCircleGray() {
    if (!fbo_) return QImage();

    context_.makeCurrent(&surface_);
    fbo_->bind();
    glViewport(0, 0, size_, size_);
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
        s.tex->setWrapMode(QOpenGLTexture::ClampToEdge);
      }
      program_->setUniformValue("minAngle", s.minA);
      program_->setUniformValue("maxAngle", s.maxA);
      s.tex->bind(0);
      glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    fbo_->release();
    QImage rgba = fbo_->toImage();  // 일반적으로 ARGB32
    context_.doneCurrent();

    // 안전하게 Qt에서 그레이스케일로 변환
    return rgba.convertToFormat(QImage::Format_Grayscale8);
  }

  void clear() {
    context_.makeCurrent(&surface_);
    for (auto& s : sectors_) {
      if (s.tex) { delete s.tex; s.tex = nullptr; }
    }
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

// ------------------ main ------------------
int main(int argc, char** argv)
{
  // Qt 오프스크린
  qputenv("QT_QPA_PLATFORM","offscreen");
  qputenv("QT_OPENGL",     "software");

  ros::init(argc, argv, "radar_img_node");
  QGuiApplication app(argc, argv);
  ros::NodeHandle nh("~");

  // 파라미터
  std::string input_topic; nh.param("input_topic", input_topic, std::string("/HaloA/data"));
  std::string ctrl_topic  = "/HaloA/change_state";
  std::string compression_format; nh.param("compression_format", compression_format, std::string("png")); // "png" or "jpeg"
  int jpeg_quality; nh.param("jpeg_quality", jpeg_quality, 90); // 0~100
  std::string frame_id; nh.param("frame_id", frame_id, std::string("map"));

  // 레이더 전송 상태
  sendRadarStatusSync(nh, ctrl_topic, "transmit");

  ros::Publisher img_pub = nh.advertise<sensor_msgs::CompressedImage>("radar_image/compressed",1);
  ros::Publisher range_pub =
      nh.advertise<marine_radar_control_msgs::RadarControlValue>(ctrl_topic,1,true);

  OffscreenRadarRenderer renderer;
  if (!renderer.init()) { ROS_ERROR("OpenGL init failed"); return -1; }

  float last = 0.0f; bool first = true;
  const float WRAP = static_cast<float>(M_PI);
  bool fbo_ready = false;

  ros::Subscriber sector_sub =
    nh.subscribe<marine_sensor_msgs::RadarSector>(input_topic, 10,
      [&](const marine_sensor_msgs::RadarSector::ConstPtr& msg){
        // 최초 수신 시 FBO 크기 결정 (지름 = echo 수 × 2)
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
          // 렌더 → 그레이스케일 이미지
          QImage gray_q = renderer.renderFullCircleGray();

          // QImage(Grayscale8) → cv::Mat (CV_8UC1)
          cv::Mat gray(gray_q.height(), gray_q.width(), CV_8UC1,
                       const_cast<uchar*>(gray_q.bits()), gray_q.bytesPerLine());
          // 인코딩을 위해 연속 메모리 확보
          cv::Mat gray_contig = gray.clone();

          // CompressedImage로 인코딩/퍼블리시
          std_msgs::Header h; 
          h.stamp   = ros::Time::now(); 
          h.frame_id= frame_id;

          std::vector<uchar> buf;
          sensor_msgs::CompressedImage cmsg;
          cmsg.header = h;

          bool ok = false;
          if (compression_format == "jpeg" || compression_format == "jpg") {
            int q = std::max(0, std::min(100, jpeg_quality));
            std::vector<int> params;
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(q);
            ok = cv::imencode(".jpg", gray_contig, buf, params);
            cmsg.format = "jpeg";
          } else {
            // png (무손실)
            ok = cv::imencode(".png", gray_contig, buf);
            cmsg.format = "png";
          }

          if (!ok) {
            ROS_ERROR("cv::imencode() failed (format=%s)", cmsg.format.c_str());
            return; // 이번 프레임은 스킵
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
