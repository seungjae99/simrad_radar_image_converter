#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <marine_sensor_msgs/RadarSector.h>
#include <sensor_msgs/Image.h>

#include <QGuiApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QTimer>
#include <QDateTime>
#include <QImage>
#include <deque>

static const char *VERT_SHADER =
    "#version 330 core\n"
    "layout(location=0) in vec4 vertex;\n"
    "out vec2 texc;\n"
    "uniform mat4 matrix;\n"
    "void main() {\n"
    "  gl_Position = matrix * vertex;\n"
    "  texc = vertex.xy;\n"
    "}\n";

static const char *FRAG_SHADER =
    "#version 330 core\n"
    "in vec2 texc;\n"
    "out vec4 fragColor;\n"
    "uniform sampler2D texture_sampler;\n"
    "uniform float minAngle, maxAngle, opacity;\n"
    "const float M_PI = 3.14159265358979323846;\n"
    "void main() {\n"
    "  if(texc.x==0.0) discard;\n"
    "  float r = length(texc);\n"
    "  if(r>1.0) discard;\n"
    "  float theta = atan(-texc.x, texc.y);\n"
    "  if(minAngle>0.0 && theta<0.0) theta+=2.0*M_PI;\n"
    "  if(theta<minAngle||theta>maxAngle) discard;\n"
    "  vec2 uv = vec2(r, (theta-minAngle)/(maxAngle-minAngle));\n"
    "  fragColor = opacity * texture(texture_sampler, uv);\n"
    "}\n";

struct Sector {
  QImage image;
  float minA, maxA;
  QDateTime timestamp;
  QOpenGLTexture* tex = nullptr;
};

class OffscreenRadarRenderer : protected QOpenGLFunctions {
public:
  OffscreenRadarRenderer(int size, double fade_time)
    : size_(size), fade_time_(fade_time),
      surface_(), context_(), fbo_(nullptr),
      program_(nullptr), quadVBO_(0)
  {}

  bool init() {
    QSurfaceFormat fmt;
    fmt.setVersion(3,3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    surface_.setFormat(fmt);
    surface_.create();
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
    QMatrix4x4 mat; mat.setToIdentity();
    program_->setUniformValue("matrix", mat);
    program_->setUniformValue("texture_sampler", 0);

    GLfloat verts[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
    glGenBuffers(1,&quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    QOpenGLFramebufferObjectFormat ffmt;
    ffmt.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
    fbo_ = new QOpenGLFramebufferObject(size_, size_, ffmt);

    context_.doneCurrent();
    return true;
  }

  void addSector(const marine_sensor_msgs::RadarSector& msg) {
    int H = msg.intensities.size();
    int W = msg.intensities.front().echoes.size();
    QImage polar(W,H,QImage::Format_Grayscale8);
    polar.fill(Qt::darkGray);
    for(int i=0;i<H;i++)
      for(int j=0;j<W;j++)
        polar.bits()[(H-1-i)*W + j] = uchar(msg.intensities[i].echoes[j]*255);

    float a1 = msg.angle_start;
    float a2 = a1 + msg.angle_increment*(H-1);
    float half = (a2-a1)/(2.0f*H);
    float minA = a2 - half*1.1f;
    float maxA = a1 + half*1.1f;

    sectors_.push_back({polar, minA, maxA, QDateTime::currentDateTimeUtc(), nullptr});
  }

  QImage renderComposite() {
    context_.makeCurrent(&surface_);
    fbo_->bind();
    glViewport(0,0,size_,size_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto now = QDateTime::currentDateTimeUtc();
    while(!sectors_.empty() && 
          sectors_.front().timestamp.msecsTo(now)/1000.0 > fade_time_) {
      delete sectors_.front().tex;
      sectors_.pop_front();
    }

    for(auto &s: sectors_) {
      if(!s.tex) {
        s.tex = new QOpenGLTexture(s.image);
        s.tex->setMinificationFilter(QOpenGLTexture::Linear);
        s.tex->setMagnificationFilter(QOpenGLTexture::Linear);
      }
      float age = s.timestamp.msecsTo(now)/1000.0f;
      float frac = float(age / fade_time_);
      float opacity = 1.0f - qBound(0.0f, frac, 1.0f);

      program_->setUniformValue("minAngle", s.minA);
      program_->setUniformValue("maxAngle", s.maxA);
      program_->setUniformValue("opacity", opacity);

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

private:
  int size_;
  double fade_time_;
  QOffscreenSurface   surface_;
  QOpenGLContext      context_;
  QOpenGLFramebufferObject* fbo_;
  QOpenGLShaderProgram*     program_;
  GLuint quadVBO_;
  std::deque<Sector> sectors_;
};

int main(int argc, char** argv) {
  qputenv("QT_QPA_PLATFORM", "offscreen");
  qputenv("QT_OPENGL", "software");

  ros::init(argc, argv, "radar_offscreen_node");
  QGuiApplication app(argc, argv);

  ros::NodeHandle nh("~");
  std::string topic;
  nh.param<std::string>("input_topic", topic, "/HaloA/data");

  image_transport::ImageTransport it(nh);
  auto pub = it.advertise("radar_image_offscreen", 1);

  OffscreenRadarRenderer renderer(512, 5.0);
  if(!renderer.init()){
    ROS_ERROR("Offscreen GL 초기화 실패");
    return -1;
  }

  ros::Subscriber sub = nh.subscribe<marine_sensor_msgs::RadarSector>(
    topic, 10, [&](auto msg){ renderer.addSector(*msg); });

  QTimer pubTimer;
  QObject::connect(&pubTimer, &QTimer::timeout, [&](){
    QImage img = renderer.renderComposite();
    cv::Mat mat(img.height(), img.width(), CV_8UC4,
                const_cast<uchar*>(img.bits()), img.bytesPerLine());
    cv::Mat bgr, gray;
    cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std_msgs::Header hdr;
    hdr.stamp = ros::Time::now();
    auto out = cv_bridge::CvImage(hdr, "mono8", gray).toImageMsg();
    pub.publish(out);
  });
  pubTimer.start(100);

  QTimer rosTimer;
  QObject::connect(&rosTimer, &QTimer::timeout, [&](){
    if(!ros::ok()) app.quit();
    else ros::spinOnce();
  });
  rosTimer.start(10);

  return app.exec();
}