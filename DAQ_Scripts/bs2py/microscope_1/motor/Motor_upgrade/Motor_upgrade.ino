//define pin numbers
#define IN1  8 // old: 8
#define IN2  9 // old: 9
#define IN3  10 // old: 10
#define IN4  11 // old: 11
//#define R 1.73814 //Microscope dial radius in cm
#define R 1.68 //Microscope dial radius in cm
#define R2 1.68 //Microscope dial radius in cm
#define r 1.0 //Gear radius in cm

//for the delay
int mils = 2;
int angle = 0;
float k = R/r;  //For the gear sync
float k2 = R2/r;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  Serial.begin(9600);
}

void loop() {  
  // Read input from serial Monitor if available
  if (Serial.available()) {
    angle = Serial.parseInt();
    rotate(angle);
  }
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
