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
#define R 1.68 //Microscope dial radius in cm
#define R2 1.68 //Microscope dial radius in cm
#define r 1.0 //Gear radius in cm

#define SL_R 1 // slider motor gear pitch radius, cm
#define SL_module 0.1 // slider motor gear module, cm/tooth
#define dist 2.9 // distance to move slider, cm

//for the delay
int mils = 2;
int angle = 0;
float k = R/r;  //For the gear sync
float k2 = R2/r;

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
  Serial.begin(9600);
}

void loop() {  
  // Read input from serial Monitor if available
  if (Serial.available()) {
    //angle = Serial.parseInt();
    mode = Serial.parseInt();
    // if the mode has been updated, move the motor
    if (mode != current_mode){
      rotate_to(mode,current_mode);    
    }
    // if moving to dark field from somewhere else, disengage the slider
    if ((mode == 2) && (current_mode != 2)) {
      slider_disengage();
    }
    // if moving to something else from dark field, engage the slider
    if ((mode != 2) && (current_mode == 2)) {
      slider_engage();
    }
    // update the mode
    current_mode = mode;
  }
}

void slider_engage(void) {
  angle = -180/3.14159*(dist/SL_R);
  rotate_slider(angle);
}

void slider_disengage(void) {
  angle = 180/3.14159*(dist/SL_R);
  rotate_slider(angle);
}

// from the mode, determine the angle to move the motor
void rotate_to(int mode, int current_mode) {
  // calculate angle to rotate
  angle = -90*(mode-current_mode);
  rotate(angle);
}

void rotate(int angle)
{
  int numberOfRounds;
  if (angle>0){
    numberOfRounds = (int) (k * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
  }
  else{
    numberOfRounds = (int) (k2 * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
  }
  if (numberOfRounds>0){
    for (int i = 0; i < numberOfRounds; i++) {
  
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
    }
}
else{
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
      
      
      
  }
}

  // Switch off all phases
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void rotate_slider(int angle)
{
  int numberOfRounds;
  if (angle>0){
    numberOfRounds = (int) (k * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
  }
  else{
    numberOfRounds = (int) (k2 * angle * 1.42222); // k*(angle_of_Micro_Dial)*(512/360)
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
