"""Raster interface for pretrained SANA column arithmetic; no solver at inference."""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL='Efficient-Large-Model/Sana_600M_512px_diffusers'
REVISION='83d7a190bfd1fd070570a793d2dab5c7a3231b9d'
CARRY_IN=(336,48,432,208)
CARRY_OUT=(144,320,240,480)
DIGIT_OUT=(272,320,368,480)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()

def render(a,b,c,font,variant=0,copy=False,answer=None,carry_pixels=None):
 im=Image.new('RGB',(512,512),'white'); d=ImageDraw.Draw(im)
 f=ImageFont.truetype(str(font),104,layout_engine=ImageFont.Layout.BASIC)
 small=ImageFont.truetype(str(font),24,layout_engine=ImageFont.Layout.BASIC)
 dx,dy=[(0,0),(-3,2),(3,-2),(1,3)][variant%4]
 d.text((256,28),'COPY' if copy else 'ADD',font=small,fill='black',anchor='mm')
 for x,value in [(128,a),(256,b),(384,c)]:
  if copy and x!=128:continue
  d.text((x+dx,128+dy),str(value),font=f,fill='black',anchor='mm')
 if not copy:
  d.text((192,128),'+',font=small,fill='black',anchor='mm')
  d.text((320,128),'+',font=small,fill='black',anchor='mm')
 d.line((80,256,432,256),fill='black',width=3)
 if carry_pixels is not None:
  assert carry_pixels.size==(96,160)
  im.paste(carry_pixels,CARRY_IN[:2])
 if answer is not None:
  for x,value in [(192,answer//10),(320,answer%10)]:
   d.text((x,400),str(value),font=f,fill='black',anchor='mm')
 return im

def pixels(im):return np.asarray(im,dtype=np.float32).transpose(2,0,1)/127.5-1

def pil(x):return Image.fromarray(np.clip((x.transpose(1,2,0)+1)*127.5,0,255).round().astype('uint8'))

def answer_templates(font):
 return [np.asarray(render(0,0,0,font,answer=i).crop(DIGIT_OUT).convert('L'),dtype=np.float32)/255 for i in range(10)]

def read_digit(im,box,templates):
 x=np.asarray(im.crop(box).convert('L'),dtype=np.float32)/255
 # Every label uses the same fixed crop; reject malformed shapes rather than always argmin.
 distances=[float(np.mean((x-t)**2)) for t in templates]
 pred=int(np.argmin(distances)); ink=float(np.mean(x<0.5))
 return {'prediction':pred,'mse':distances[pred],'ink':ink,'valid':distances[pred]<0.055 and 0.015<ink<0.5}

def read_answer(im,templates):
 a=read_digit(im,CARRY_OUT,templates);b=read_digit(im,DIGIT_OUT,templates)
 return {'prediction':10*a['prediction']+b['prediction'],'valid':a['valid'] and b['valid'] and a['prediction']<=1,'carry':a,'digit':b}

def atomic_json(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)
