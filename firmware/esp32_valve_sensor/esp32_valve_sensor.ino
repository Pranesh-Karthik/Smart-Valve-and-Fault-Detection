// Optional acquisition firmware; calibration and hardware validation required.
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include "secrets.h"

constexpr int CT_PIN=34, PRESSURE_PIN=35, N=1000, UDP_PORT=5005;
constexpr float CT_MIDPOINT=1900, CT_CALIB_FACTOR=.0488f;
constexpr float PRESSURE_MAX_BAR=10.0f;
float samples[N];
WiFiUDP udp;
String label="unlabeled", command="";

bool validLabel(const String &s) {
  return s=="all_closed" || s=="valve1_open" || s=="valve2_open" ||
         s=="valve3_open" || s=="leak_detected";
}
float magnitude(int hz) {
  float re=0, im=0;
  for(int n=0;n<N;n++) {
    float angle=2.0f*PI*hz*n/1000.0f;
    re+=samples[n]*cosf(angle); im-=samples[n]*sinf(angle);
  }
  return sqrtf(re*re+im*im)/N;
}
float bandEnergy(int center) {
  float e=0;
  for(int hz=center-5;hz<=center+5;hz++){float m=magnitude(hz);e+=m*m;}
  return e;
}
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  WiFi.begin(WIFI_SSID,WIFI_PASS); // Acquisition does not wait indefinitely for Wi-Fi.
  udp.begin(UDP_PORT);
}
void loop() {
  while(Serial.available()) {
    char ch=Serial.read();
    if(ch=='\n') {
      command.trim();
      if(command.startsWith("L:") && validLabel(command.substring(2))) label=command.substring(2);
      command="";
    } else if(command.length()<80) command+=ch;
  }
  uint32_t deadline=micros(), pressureSum=0;
  int late=0;
  for(int i=0;i<N;i++) {
    while((int32_t)(micros()-deadline)<0) {} // Handles micros() wraparound.
    if((int32_t)(micros()-deadline)>250) late++;
    samples[i]=(analogRead(CT_PIN)-CT_MIDPOINT)*CT_CALIB_FACTOR;
    if(i%10==0) pressureSum+=analogRead(PRESSURE_PIN);
    deadline+=1000;
  }
  if(late) { Serial.println("# Dropped window: sampling deadlines missed"); return; }
  float pressure=(pressureSum/100.0f)*(PRESSURE_MAX_BAR/4095.0f);
  float sq=0, absSum=0, sum=0, peak=0, variance=0;
  for(int i=0;i<N;i++) {
    float x=samples[i]; sq+=x*x; absSum+=fabsf(x); sum+=x;
    if(fabsf(x)>peak) peak=fabsf(x);
  }
  float mean=sum/N, rms=sqrtf(sq/N);
  for(int i=0;i<N;i++){float d=samples[i]-mean;variance+=d*d;}
  variance/=N;
  float domFreq=40, domMag=-1;
  for(int hz=40;hz<=300;hz+=5) {
    float m=magnitude(hz);
    if(m>domMag){domFreq=hz;domMag=m;}
  }
  float h3=bandEnergy(150), h5=bandEnergy(250);
  if(!validLabel(label)){Serial.println("# Set label using L:<label>");return;}
  char row[512];
  int length=snprintf(row,sizeof(row),
    "%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%s",
    rms,absSum/N,sqrtf(variance),peak,rms>1e-9f?peak/rms:0,variance,
    domFreq,domMag,h3,h5,sq,pressure,label.c_str());
  if(length<0 || length>=(int)sizeof(row)){Serial.println("# Row overflow");return;}
  Serial.println(row);
  if(WiFi.status()==WL_CONNECTED) {
    udp.beginPacket(LAPTOP_IP,UDP_PORT);udp.print(row);udp.endPacket();
  }
}
