#include <Adafruit_LIS2MDL.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
Adafruit_LIS2MDL lis2mdl = Adafruit_LIS2MDL(12345);

//define pin numbers: cube motor
#define IN1  8
#define IN2  9
#define IN3  10
#define IN4  11

// define pin numbers: slider motor
#define SL_IN1  2
#define SL_IN2  3
#define SL_IN3  4
#define SL_IN4  5

//#define R 1.73814 //Microscope dial radius in cm
#define R 2.0 //Microscope dial radius in cm for rotation to the left (motor CW)
#define R2 1.68 //Microscope dial radius in cm for rotation to the right (motor CCW)
#define r 1.0 //Gear radius in cm

#define SL_R 1 // slider motor gear pitch radius, cm
#define SL_module 0.1 // slider motor gear module, cm/tooth
#define dist 2.9 // distance to move slider, cm

//for the delay
//int MILSVAL = 2;
int mils = 2;                //stepper motor winding delay
int angle = 0;
float k = R/r;  //For the gear sync
float k2 = R2/r;

int zero_cross_neg;         // keep track of when magnetic field falls below zero
int zero_cross_pos;         // keep track of when magnetic field rises above zero
int magnet_crossing;        // keep track of how many magnets we pass
int lo_Bz_trip_level = -100; // keep track of when magnetic field passes below this value
int hi_Bz_trip_level = 2500;// keep track of when magnetic field passes above this value
bool lo_Bz_trip;            // whether or not it did
bool hi_Bz_trip; 

int mode = 3; // what mode the microscope is in:
              // 1: bright field
              // 2: dark field
              // 3: photoluminescence
              // 4: currently unused
int current_mode = 3;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(SL_IN1, OUTPUT);
  pinMode(SL_IN2, OUTPUT);
  pinMode(SL_IN3, OUTPUT);
  pinMode(SL_IN4, OUTPUT);
  Serial.begin(115200);

  while (!Serial)
  delay(10); // will pause Zero, Leonardo, etc until serial console opens

  /* Enable auto-gain */
  lis2mdl.enableAutoRange(true);

  /* Initialise the sensor */
  if (!lis2mdl.begin()) {  // I2C mode
  //if (! lis2mdl.begin_SPI(LIS2MDL_CS)) {  // hardware SPI mode
  //if (! lis2mdl.begin_SPI(LIS2MDL_CS, LIS2MDL_CLK, LIS2MDL_MISO, LIS2MDL_MOSI)) { // soft SPI
    /* There was a problem detecting the LIS2MDL ... check your connections */
//    Serial.println("Ooops, no LIS2MDL detected ... Check your wiring!");
    while (1) delay(10);
  } 
//  else {
//    Serial.println("We have a problem");
//  }

//  Serial.println("Sensor Initialized");
//  Serial.println("");
//  
  /* Display some basic information on this sensor */
  lis2mdl.printSensorDetails();
}

void loop() {  
  sensors_event_t event;
  lis2mdl.getEvent(&event);
  int current_mag_Z = event.magnetic.z;
  String stringBz = String(current_mag_Z);
  
  // Read input from serial Monitor if available
  if (Serial.available()) {
    //angle = Serial.parseInt();
    mode = Serial.parseInt();
    // if the mode has been updated, move the motor
    if (mode != current_mode){
      rotate_to(mode,current_mode);    
    }
//    // if moving to dark field from somewhere else, disengage the slider
//    if ((mode == 2) && (current_mode != 2)) {
//      slider_disengage();
//    }
//    // if moving to something else from dark field, engage the slider
//    if ((mode != 2) && (current_mode == 2)) {
//      slider_engage();
//    }
    // update the mode
    current_mode = mode;
  }
}

//void slider_engage(void) {
//  angle = -180/3.14159*(dist/SL_R);
//  rotate_slider(angle);
//}
//
//void slider_disengage(void) {
//  angle = 180/3.14159*(dist/SL_R);
//  rotate_slider(angle);
//}

// from the mode, determine the angle to move the motor
void rotate_to(int mode, int current_mode) {
  sensors_event_t event;
  int max_mag_Z = 4600;
  int current_mag_Z = 0;
  int step_angle = 10;                          //1 degree fine step
  bool hit_max = false;                         //SET boolean for hit_max
  int state = current_mode;
  //magnet_crossing = 0;
  mils = 2;  // delay time between activating each combination of windings in the stepper motor
  
  //calculate angle to rotate
  angle = -90*(mode-state);
  int sign_angle = angle/abs(angle);            //determine sign for rotation

  zero_cross_neg = 0;                           //# of times motor crossed 0 from positive -> negative
  zero_cross_pos = 0;                           //# of times motor crossed 0 from negative -> positive

  lo_Bz_trip = false;                           // did we pass through a minimum
  hi_Bz_trip = false;                           // did we reach another magnet after passing through a minimum
  
  while(state!=mode){
    magnet_crossing = 0;                          // how many magnets we passed
    delay(1000);
    
    //initial rotation
    mils = 2;
    hit_max = rotate(-90*sign_angle,true);         //coarse rotate but only until mag values pass lower threshold (currently 3000)
    delay(1000);
    
    //print info on current mag value before final rotation
    lis2mdl.getEvent(&event);                     //call mag sensor for value
    current_mag_Z = event.magnetic.z;
    Serial.print("<Before hit max code Mag Z: values: ");
    Serial.print(current_mag_Z);
    Serial.print("\n");         
    Serial.print("Final 'fine' rotate ");
    Serial.print("\n");    
    Serial.print(">");
    
    Serial.print("<Motor finished rotations, attempt to complete rotation>");
    while (abs(current_mag_Z) < max_mag_Z){       //
      rotate(-1.5*sign_angle,false);
      lis2mdl.getEvent(&event);
      current_mag_Z = event.magnetic.z;           
      }
      
    if (current_mag_Z > max_mag_Z) {
      Serial.print("<Found Home>");
      }
    Serial.print("<Hit Max? ");
    Serial.print(hit_max);
    Serial.print(">");
   
    //state = ((state - 1 - sign_angle*magnet_crossing)+4)%4 + 1;  //chcek the actual state of the filter cube wheel
    state = state - sign_angle*magnet_crossing;
    if (state <= 0){
      state = 4 + state;
    }
    if (state >= 5){
      state = state - 4;
    }
    
    Serial.print("<");                                         // and report to Python
    Serial.print("\n");  
    Serial.print("Zero Crossings: ");
    Serial.print(zero_cross_pos);
    Serial.print("  ");
    Serial.print("Magnet Crossings: ");
    Serial.print(magnet_crossing);
    Serial.print("\n");
    Serial.print("Desired Mode: ");
    Serial.print(mode);
    Serial.print("\n");
    Serial.print("Actual Mode: ");
    Serial.print(state);
    Serial.print("\n");
    Serial.print(">");

     // if at the wrong mode, return to the right place:     

    if (state != mode) {
      Serial.print("<At wrong mode! Rotate to desired mode>");   
      }
    else {
      Serial.print("<Finished>");
      }
    }
}

bool rotate(int angle, bool coarse)
{
  int numberOfRounds;
  int curr_mag_Z;
  int last_mag_Z;
  int max_mag_Z; 
  
  sensors_event_t event;
  bool went_thru_max = false;
  
  if (angle>0) {                                    // rotates from larger number to smaller number (to the left)
    numberOfRounds = (int) (k * angle * 1.42222);   // k*(angle_of_Micro_Dial)*(512/360)
  }
  else{                                             // rotates from smaller number to larger number (to the right)
    numberOfRounds = (int) (2.0 * angle * 1.42222);  // k*(angle_of_Micro_Dial)*(512/360)
  }
  if (numberOfRounds>0){                            //# of rotations to the left
    max_mag_Z = 2500;
    for (int i = 1; i < numberOfRounds+1; i++) {      //changes poles of motor to rotate through total number of rounds
      //1000                                        
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      delay(mils);
  
      //1100
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      delay(mils);
  
      //0100
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      delay(mils);
  
      //0110
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      delay(mils);
  
      //0010
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      delay(mils);
  
      //0011
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, HIGH);
      delay(mils);
  
      //0001
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      delay(mils);
  
      //1001
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      delay(mils);

      lis2mdl.getEvent(&event);
      last_mag_Z = curr_mag_Z;
      curr_mag_Z = event.magnetic.z;
      Serial.print("<");
      //Serial.print("<Last Mag Z values: ");
      //Serial.print(last_mag_Z);
      //Serial.print("   ");
      Serial.print("Mag Z values: ");
      Serial.print(curr_mag_Z);
      Serial.print(" uT");
      Serial.print(">");
      
      // when the magnetic field drops below zero, note it
      if ((curr_mag_Z < 0) && (last_mag_Z > 0)){
        zero_cross_neg++;
        //Serial.print("<Just crossed Bz = 0>");
      }

      // when the magnetic field rises above zero, note it
      if ((curr_mag_Z >= 0) && (last_mag_Z < 0)){
        //Serial.print("<Just crossed Bz = 0>");
        zero_cross_pos++;
      }
      
      // detect if we pass through a minimum
      if (!lo_Bz_trip && !hi_Bz_trip){
        if (curr_mag_Z < lo_Bz_trip_level){
          lo_Bz_trip = true;
          Serial.print("<Tripped low!>");  
        }
      }
      
      // detect if we get near the next magnet after passing thru a minimum
      if (lo_Bz_trip && !hi_Bz_trip){
        if (curr_mag_Z > hi_Bz_trip_level){
          hi_Bz_trip = true;
          Serial.print("<Tripped high!>");   
        }
      }

      // if both high and low limits have been tripped, reset them and increment the magnet counter
      if (lo_Bz_trip && hi_Bz_trip){
        magnet_crossing++;
        lo_Bz_trip = false;
        hi_Bz_trip = false;
      }
      
      // if we're getting close to the next position, start logging the field
      if (i>0.7*abs(numberOfRounds)){                //determine if wheel is past a certain point
        if (coarse){
          mils = 10; // slow down!
        } 
        if (abs(curr_mag_Z) > max_mag_Z) {            //Check if mag sensor has detected magnet
         
          went_thru_max = true;                       //If went_thru_max is true, then hit_max is true and rotate_to() function will finish....

          // if the high trip didn't happen but we're at the magnet, note it and reset the trips 
          //if (lo_Bz_trip && !hi_Bz_trip){
          //  magnet_crossing++;
          //  lo_Bz_trip = false;
          //  hi_Bz_trip = false; 
          //}
          
          digitalWrite(IN1, LOW);
          digitalWrite(IN2, LOW);
          digitalWrite(IN3, LOW);
          digitalWrite(IN4, LOW);
          
          return went_thru_max;
        }                                             //(as long as current mag > max mag)
      }
    }
      
}
else{                                                 //# of rotations to the right
  max_mag_Z = 2500;
  for (int i = 0; i < abs(numberOfRounds); i++) {
  
      //1000
      digitalWrite(IN4, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN1, LOW);
      delay(mils);

      //1100
      digitalWrite(IN4, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN1, LOW);
      delay(mils);

      //0100
      digitalWrite(IN4, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN1, LOW);
      delay(mils);
      
      //0110
      digitalWrite(IN4, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN1, LOW);
      delay(mils);

      //0010
      digitalWrite(IN4, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN1, LOW);
      delay(mils);

      //0011
      digitalWrite(IN4, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN1, HIGH);
      delay(mils);

      //0001
      digitalWrite(IN4, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN1, HIGH);
      delay(mils);
     
     //1001
      digitalWrite(IN4, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN1, HIGH);
      delay(mils); 
        
      lis2mdl.getEvent(&event);
      last_mag_Z = curr_mag_Z;
      curr_mag_Z = event.magnetic.z;
      Serial.print("<");
      //Serial.print("<Last Mag Z values: ");
      //Serial.print(last_mag_Z);
      //Serial.print("   ");
      Serial.print("Mag Z values: ");
      Serial.print(curr_mag_Z);
      Serial.print(" uT");
      Serial.print(">");

      // when the magnetic field drops below zero, note it
      if ((curr_mag_Z < 0) && (last_mag_Z > 0)){
        zero_cross_neg++;
        //Serial.print("<Just crossed Bz = 0>");
      }

      // when the magnetic field rises above zero, note it
      if ((curr_mag_Z >= 0) && (last_mag_Z < 0)){
        //Serial.print("<Just crossed Bz = 0>");
        zero_cross_pos++;
      }

      // detect if we pass through a minimum
      if (!lo_Bz_trip && !hi_Bz_trip){
        if (curr_mag_Z < lo_Bz_trip_level){
          lo_Bz_trip = true;
          Serial.print("<Tripped low!>");  
        }
      }
      
      // detect if we get near the next magnet after passing thru a minimum
      if (lo_Bz_trip && !hi_Bz_trip){
        if (curr_mag_Z > hi_Bz_trip_level){
          hi_Bz_trip = true;
          Serial.print("<Tripped high!>");  
        }
      }

      // if both high and low limits have been tripped, reset them and increment the magnet counter
      if (lo_Bz_trip && hi_Bz_trip){
        magnet_crossing++;
        lo_Bz_trip = false;
        hi_Bz_trip = false;
      }

      // if we're getting close to the next position, start logging the field
      if (i>0.75*abs(numberOfRounds)){
        if (coarse){
          mils = 15; // slow down!
        }                                   // slow down!
        if (abs(curr_mag_Z) > max_mag_Z) {
          went_thru_max = true;
          digitalWrite(IN1, LOW);
          digitalWrite(IN2, LOW);
          digitalWrite(IN3, LOW);
          digitalWrite(IN4, LOW);
          return went_thru_max;
        }
      }
  }
}
// Switch off all phases
digitalWrite(IN1, LOW);
digitalWrite(IN2, LOW);
digitalWrite(IN3, LOW);
digitalWrite(IN4, LOW);

return went_thru_max;

}



void rotate_slider(int angle)
{
  int numberOfRounds;
  mils = 2;
  if (angle>0){
    numberOfRounds = (int) (1.45 * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
  }
  else{
    numberOfRounds = (int) (1.45 * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
  }
  if (numberOfRounds>0){
    for (int i = 0; i < numberOfRounds; i++) {
  
      //1000
      digitalWrite(SL_IN1, HIGH);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN4, LOW);
      delay(mils);
  
      //1100
      digitalWrite(SL_IN1, HIGH);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN4, LOW);
      delay(mils);
  
      //0100
      digitalWrite(SL_IN1, LOW);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN4, LOW);
      delay(mils);
  
      //0110
      digitalWrite(SL_IN1, LOW);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN4, LOW);
      delay(mils);
  
      //0010
      digitalWrite(SL_IN1, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN4, LOW);
      delay(mils);
  
      //0011
      digitalWrite(SL_IN1, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN4, HIGH);
      delay(mils);
  
      //0001
      digitalWrite(SL_IN1, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN4, HIGH);
      delay(mils);
  
      //1001
      digitalWrite(SL_IN1, HIGH);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN4, HIGH);
      delay(mils);
    }
}
else{
  for (int i = 0; i < abs(numberOfRounds); i++) {
  
      //1000
      digitalWrite(SL_IN4, HIGH);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN1, LOW);
      delay(mils);

      //1100
      digitalWrite(SL_IN4, HIGH);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN1, LOW);
      delay(mils);

      //0100
      digitalWrite(SL_IN4, LOW);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN1, LOW);
      delay(mils);
      
      //0110
      digitalWrite(SL_IN4, LOW);
      digitalWrite(SL_IN3, HIGH);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN1, LOW);
      delay(mils);

      //0010
      digitalWrite(SL_IN4, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN1, LOW);
      delay(mils);

      //0011
      digitalWrite(SL_IN4, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN2, HIGH);
      digitalWrite(SL_IN1, HIGH);
      delay(mils);

      //0001
      digitalWrite(SL_IN4, LOW);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN1, HIGH);
      delay(mils);
     
     //1001
      digitalWrite(SL_IN4, HIGH);
      digitalWrite(SL_IN3, LOW);
      digitalWrite(SL_IN2, LOW);
      digitalWrite(SL_IN1, HIGH);
      delay(mils); 
      
      
      
  }
}

  // Switch off all phases
  digitalWrite(SL_IN1, LOW);
  digitalWrite(SL_IN2, LOW);
  digitalWrite(SL_IN3, LOW);
  digitalWrite(SL_IN4, LOW);
}
