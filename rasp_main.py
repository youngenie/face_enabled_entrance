#!/usr/bin/python3
import RPi.GPIO as GPIO
import picamera
import time
import os
from pygame import mixer
import Adafruit_CharLCD as LCD

# Raspberry Pi pin configuration:
lcd_rs        = 25
lcd_en        = 24
lcd_d4        = 23
lcd_d5        = 17
lcd_d6        = 27
lcd_d7        = 22
lcd_backlight = 2

# Define LCD column and row size for 16x2 LCD.
lcd_columns = 16
lcd_rows    = 2

# Initialize the LCD using the pins above.
lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7,
                           lcd_columns, lcd_rows, lcd_backlight)

GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN)

lst = []
os.chdir('/home/lotte/sh/pics')

for item in os.listdir('/home/lotte/sh/pics'):
    if not item.startswith('.') and os.path.isfile(os.path.join('/home/lotte/sh/pics',item)):
        comp = int(item.split('.')[0])
        lst.append(comp)

cnt = sorted(lst)[-1] + 1
cnt

def screen_display():
    for i in range(lcd_columns-len(message)):
        time.sleep(0.5)
        lcd.move_right()
    for i in range(lcd_columns-len(message)):
        time.sleep(0.5)
        lcd.move_left()

#Start Message
lcd.message('     LOTTE      \n  BigData Team  ')

while True:
    lcd.clear()
    lcd.message('     LOTTE      \n  BigData Team  ')
    lst = []
    for item in os.listdir('/home/lotte/sh/audio'):
        if not item.startswith('.') and os.path.isfile(os.path.join('/home/lotte/sh/audio', item)):
            lst.append(item)
    print("True?")
    if len(lst) == 2:
        time.sleep(0.5)
        if GPIO.input(21):
            
            print("object detected")
            with picamera.PiCamera() as camera:
                camera.resolution = (1024, 768)
                camera.start_preview()
                time.sleep(1)
                camera.stop_preview()
                
                mixer.init()
                mixer.music.load('/home/lotte/sh/audio/shutter.mp3')
                mixer.music.play()
                
                camera.capture(str(cnt)+ '.jpg')
                time.sleep(1)
                os.remove(str(cnt-10)+ '.jpg')
                cnt = cnt+1
                
                mixer.music.load('/home/lotte/sh/audio/bgm_36.mp3')
                mixer.music.play()
                
                #Move Message
                lcd.clear()
                message='Waiting...'
                lcd.message(message)
                
                for i in range(8):
                    screen_display()

                #End Message
                lcd.clear()
                lcd.message('   Finished!!   ')

                mixer.music.load('/home/lotte/sh/audio/tts.mp3')
                mixer.music.play()
                time.sleep(10)
                os.remove('/home/lotte/sh/audio/tts.mp3')
    else:
        os.remove('/home/lotte/sh/audio/tts.mp3')
                                        
time.sleep(3)
GPIO.cleanup()
