import numpy
import copy
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

#calcul changement d'état, si un visage est regardé ou non (plus tard)
#calcul trajectoire
#https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py

#read measures of file
def read_file(name):
    fichier=open(name, "r")
    measure=fichier.readlines()
    fichier.close()
    return measure

#get and clean all measures
def extract_measures(name):
    measure=read_file(name)
    i=0
    list_dict_measure=[]
    while(i<len(measure)):
        measure[i]=measure[i].replace("\n", "")
        measure[i]=measure[i].split("\t")
        j=0
        dict_measure={}
        while(j<len(measure[i])):
            measure[i][j]=measure[i][j].split(":")
            dict_measure[measure[i][j][0]]=measure[i][j][1]
            j=j+1
        i=i+1
        list_dict_measure.append(dict_measure)
    return list_dict_measure

#give the validity of pupil and an invalid diamater if we don't have it for one measure
def validity_pupil(direction, measure):
    dict_pupil={}
    if measure[direction+'_pupil_validity']=='1' :
        dict_pupil[direction+'_pupil_diameter']=measure[direction+'_pupil_diameter']
        dict_pupil['no_'+direction+'_pupil']=1
    else :
        dict_pupil['no_'+direction+'_pupil']=0
        dict_pupil[direction+'_pupil_diameter']='-1'
    return dict_pupil[direction+'_pupil_diameter'], dict_pupil['no_'+direction+'_pupil']

#give validity of gaze and invalid coordonates if we don't have it for one measure
def validity_gaze(direction, measure):
    dict_gaze={}
    if measure[direction+'_gaze_point_validity']=='1' :
        dict_gaze['gaze_'+direction+'_x_y']=measure[direction+'_gaze_point_on_display_area']
        dict_gaze['no_'+direction+'_gaze_x_y']=1
    else :
        dict_gaze['no_'+direction+'_gaze_x_y']=0
        dict_gaze['gaze_'+direction+'_x_y']='-1 -1'
    return dict_gaze['gaze_'+direction+'_x_y'], dict_gaze['no_'+direction+'_gaze_x_y']

#give validities of gaze and pupils for all measures
def validity(measure):
    liste_dict_features=[]
    i=0
    while(i<len(measure)):
        dict_features={}
        indict=False
        dict_features['time']=measure[i]['system_time_stamp']
        dict_features['left_pupil_diameter'], dict_features['no_left_pupil']=validity_pupil('left', measure[i])
        dict_features['right_pupil_diameter'], dict_features['no_right_pupil']=validity_pupil('right', measure[i])
        dict_features['gaze_left_x_y'], dict_features['no_left_gaze_x_y']=validity_gaze('left', measure[i])
        dict_features['gaze_right_x_y'], dict_features['no_right_gaze_x_y']=validity_gaze('right', measure[i])
        liste_dict_features.append(dict_features)
        i=i+1
    return liste_dict_features

#give number and time of blink or not blink for one eye
def blink_direction_eye(direction, measure):
    j=0
    if measure['no_'+direction+'_gaze_x_y']==0 :
        blinkbool=0
    else :
        blinkbool=1
        j=j+1
    return blinkbool


#get blink features
def blink(listmeasure):
    i=0
    while(i<len(listmeasure)):
        lefteyeblink=blink_direction_eye('left', listmeasure[i])
        righteyeblink=blink_direction_eye('right', listmeasure[i])
        listmeasure[i]['left_eye_blink']=lefteyeblink
        listmeasure[i]['right_eye_blink']=righteyeblink
        i=i+1   
    return listmeasure


#remove data of fixation or saccades blinks
def remove_missing(x, y, time, missing):
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time

#get time fixations and where in the time they happened
def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
	
	x, y, time = remove_missing(x, y, time, missing)

	# empty list to contain data
	Sfix = []
	Efix = []
	# loop through all coordinates
	si = 0
	fixstart = False
	for i in range(1,len(x)):
		# calculate Euclidean distance from the current fixation coordinate
		# to the next coordinate
		squared_distance = ((x[si]-x[i])**2 + (y[si]-y[i])**2)
		dist = 0.0
		if squared_distance > 0:
			dist = squared_distance**0.5
		# check if the next coordinate is below maximal distance
		if dist <= maxdist and not fixstart:
			# start a new fixation
			si = 0 + i
			fixstart = True
			Sfix.append([time[i]])
		elif dist > maxdist and fixstart:
			# end the current fixation
			fixstart = False
			# only store the fixation if the duration is ok
			if time[i-1]-Sfix[-1][0] >= mindur:
				Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
			# delete the last fixation start if it was too short
			else:
				Sfix.pop(-1)
			si = 0 + i
		elif not fixstart:
			si += 1
	#add last fixation end (we can lose it if dist > maxdist is false for the last point)
	if len(Sfix) > len(Efix):
		Efix.append([Sfix[-1][0], time[len(x)-1], time[len(x)-1]-Sfix[-1][0], x[si], y[si]])
	return Sfix, Efix


def zero_padding_saccade(m, maxi):
    i=0
    while(i<len(m)):
        while len(m[i][-1]['time_saccade_left'])<maxi:
            m[i][-1]['time_saccade_left'].append(0)
        while len(m[i][-1]['time_saccade_right'])<maxi:
            m[i][-1]['time_saccade_right'].append(0)
        i=i+1
    return m

#calculate time and frequencies fixations
def time_and_frequencies_fixations(m):
    i=0
    maxi=0
    maxisaccadic=0
    xleft, yleft, xright, yright, time=getxygazetime(copy.deepcopy(m))
    a_left,b_left=fixation_detection(xleft, yleft, time, missing=-1.0,mindur=50, maxdist=25)
    a_right,b_right=fixation_detection(xright, yright, time, missing=-1.0,mindur=50, maxdist=25)
    j=0
    k=0
    l=0
    while(j<len(time)):
        t=time[j]
        try :
            if t>=b_left[k][0]  and t<=b_left[k][1]:
                m[j]['fixation_left']=1
            else :
                m[j]['fixation_left']=0
            if t>b_left[k][1]:
                k=k+1
        except :
            m[j]['fixation_left']=0
        try :
            if t>=b_right[k][0]  and t<=b_right[k][1]:
                m[j]['fixation_right']=1
            else :
                m[j]['fixation_right']=0
            if t>b_right[k][1]:
                l=l+1
        except :
            m[j]['fixation_right']=0
        j=j+1

    a_left2, b_left2=saccade_detection(xleft, yleft, time, missing=-1.0, minlen=16)
    a_right2, b_right2=saccade_detection(xleft, yleft, time, missing=-1.0, minlen=16)
    j=0
    k=0
    l=0
    while(j<len(time)):
        t=time[j]
        try :
            if t>=b_left2[k][0]  and t<=b_left2[k][1]:
                m[j]['saccad_left']=1
            else :
                m[j]['saccad_left']=0
            if t>b_left2[k][1]:
                k=k+1
        except :
            m[j]['saccad_left']=0
        try :
            if t>=b_right2[k][0]  and t<=b_right2[k][1]:
                m[j]['saccad_right']=1
            else :
                m[j]['saccad_right']=0
            if t>b_right2[k][1]:
                l=l+1
        except :
            m[j]['saccad_right']=0
        j=j+1
    return  m

#detect saccads
def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
	
	x, y, time = remove_missing(x, y, time, missing)

	# CONTAINERS
	Ssac = []
	Esac = []

	# INTER-SAMPLE MEASURES
	# the distance between samples is the square root of the sum
	# of the squared horizontal and vertical interdistances
	intdist = (numpy.diff(x)**2 + numpy.diff(y)**2)**0.5
	# get inter-sample times
	inttime = numpy.diff(time)
	# recalculate inter-sample times to seconds
	inttime = inttime / 1000.0
	
	# VELOCITY AND ACCELERATION
	# the velocity between samples is the inter-sample distance
	# divided by the inter-sample time
	vel = intdist / inttime
	# the acceleration is the sample-to-sample difference in
	# eye movement velocity
	acc = numpy.diff(vel)

	# SACCADE START AND END
	t0i = 0
	stop = False
	while not stop:
		# saccade start (t1) is when the velocity or acceleration
		# surpass threshold, saccade end (t2) is when both return
		# under threshold
	
		# detect saccade starts
		sacstarts = numpy.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
		if len(sacstarts) > 0:
			# timestamp for starting position
			t1i = t0i + sacstarts[0] + 1
			if t1i >= len(time)-1:
				t1i = len(time)-2
			t1 = time[t1i]
			
			# add to saccade starts
			Ssac.append([t1])
			
			# detect saccade endings
			sacends = numpy.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
			if len(sacends) > 0:
				# timestamp for ending position
				t2i = sacends[0] + 1 + t1i + 2
				if t2i >= len(time):
					t2i = len(time)-1
				t2 = time[t2i]
				dur = t2 - t1

				# ignore saccades that did not last long enough
				if dur >= minlen:
					# add to saccade ends
					Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
				else:
					# remove last saccade start on too low duration
					Ssac.pop(-1)

				# update t0i
				t0i = 0 + t2i
			else:
				stop = True
		else:
			stop = True
	
	return Ssac, Esac


#get x, y, time for one second
def getxygazetime(m):
    j=0
    xleft=[]
    xright=[]
    yleft=[]
    yright=[]
    time=[]
    while(j<len(m)):
        xyleft=m[j]['gaze_left_x_y'].split(" ")
        xyright=m[j]['gaze_right_x_y'].split(" ")
        xleft.append(float(xyleft[0])*1000)
        yleft.append(float(xyleft[1])*1000)
        xright.append(float(xyright[0])*1000)
        yright.append(float(xyright[1])*1000)
        time.append(int(m[j]['time'])//1000)
        j=j+1
    xleft=numpy.array(xleft)
    yleft=numpy.array(yleft)
    xright=numpy.array(xright)
    yright=numpy.array(yright)
    time=numpy.array(time)
    return xleft, yleft, xright, yright, time

def distances(listmeasure):
    i=1
    listmeasure[0]['distance_from_2_points_right']=0
    listmeasure[0]['distance_from_2_points_left']=0
    xy1left=listmeasure[0]['gaze_left_x_y']
    xy1right=listmeasure[0]['gaze_right_x_y']
    while(i<len(listmeasure)):
        if listmeasure[i]['no_left_gaze_x_y']==1 :
            listmeasure[i]['distance_from_2_points_left']=dist(xy1left, listmeasure[i]['gaze_left_x_y'])
            xy1left=listmeasure[i]['gaze_left_x_y']
        else : 
            listmeasure[i]['distance_from_2_points_left']=0
        if listmeasure[i]['no_right_gaze_x_y']==1 :
            listmeasure[i]['distance_from_2_points_right']=dist(xy1right, listmeasure[i]['gaze_right_x_y'])
            xy1right=listmeasure[i]['gaze_right_x_y']
        else :
            listmeasure[i]['distance_from_2_points_right']=0
        i=i+1
    return listmeasure

def dist(m1, m2):
    d1=m1.split(" ")
    d2=m2.split(" ")
    d1[0]=float(d1[0])
    d2[0]=float(d2[0])
    d1[1]=float(d1[1])
    d2[1]=float(d2[1])
    del d1[-1]
    del d2[-1]
    try :
        dx=(d2[0]-d1[0])**2
        dy=(d2[1]-d1[1])**2
        d=math.sqrt(dx+dy)
    except :
        d=0
    return d

def angles(listmeasure):
    listmeasure[0]['angle_right']=0
    listmeasure[0]['angle_left']=0
    xy1left=listmeasure[0]['gaze_left_x_y']
    xy1right=listmeasure[0]['gaze_right_x_y']
    listmeasure[1]['angle_right']=0
    listmeasure[1]['angle_left']=0
    xy2left=listmeasure[1]['gaze_left_x_y']
    xy2right=listmeasure[1]['gaze_right_x_y']
    i=2
    while(i<len(listmeasure)):
        if listmeasure[i]['no_left_gaze_x_y']==1 :
            listmeasure[i]['angle_left']=angle(xy1left, xy2left, listmeasure[i]['gaze_left_x_y'])
            xy1left=copy.deepcopy(xy2left)
            xy2left=listmeasure[i]['gaze_left_x_y']
        else : 
            listmeasure[i]['angle_left']=0
        if listmeasure[i]['no_right_gaze_x_y']==1 :
            listmeasure[i]['angle_right']=angle(xy1right, xy2right,listmeasure[i]['gaze_right_x_y'])
            xy1right=copy.deepcopy(xy2right)
            xy2right=listmeasure[i]['gaze_right_x_y']
        else :
            listmeasure[i]['angle_right']=0
        i=i+1
    return listmeasure

def angle(m1,m2, m3):
    try :
        #c^2=a^2+b^2-2ab*cos(gamma)
        #c^2-a^2-b^2=-2ab*cos(gamma)
        #(c^2-a^2-b^2)/(-2ab)=cos(gamma)
        #arcos((c^2-a^2-b^2)/(-2ab))=gamma
        distc=dist(m3, m2)
        dista=dist(m3, m1)
        distb=dist(m2, m1)
        a=math.acos((distc**2-dista**2-distb**2)/(-2*dista*distb))
    except :
        a=0
    return a

def trajectory_one_eye(x,y):
    i=0
    x=list(x)
    y=list(y)
    while(i<len(x)):
        if x[i]<0 :
            del x[i]
            del y[i]
            i=i-1
        i=i+1
    x=numpy.array(x)
    y=numpy.array(y)
    ytest=[]
    mean_squared=[]
    i=1

    while(i<len(x)+1):
        ytest.append(polynome(x,y, i))
        mean_squared.append(mean_squared_error(y,ytest[-1]))
        i=i+1
    if mean_squared==[]:
        yt=[0]
    else :
        mean_squared=numpy.array(mean_squared)
        yt=ytest[numpy.argmin(mean_squared)]
    return yt

def polynome(x,y, deg):
    model = make_pipeline(PolynomialFeatures(deg), Ridge())
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    model.fit(x, y)
    Y_pred = model.predict(x)
    return Y_pred


def trajectory(x, y, time):
    polyx=trajectory_one_eye(time,x)
    polyy=trajectory_one_eye(time,y)
    return polyx, polyy

def get_features1(listmeasure):
    i=0
    features=[]
    while(i<len(listmeasure)):
        feature=[listmeasure[i]['time'], listmeasure[i]['left_pupil_diameter'], listmeasure[i]['right_pupil_diameter'], listmeasure[i]['gaze_left_x_y'].split(" ")[0],listmeasure[i]['gaze_left_x_y'].split(" ")[1], listmeasure[i]['gaze_right_x_y'].split(" ")[0],listmeasure[i]['gaze_right_x_y'].split(" ")[1], listmeasure[i]['left_eye_blink'], listmeasure[i]['right_eye_blink'], listmeasure[i]['fixation_left'], listmeasure[i]['fixation_right'], listmeasure[i]['saccad_left'], listmeasure[i]['saccad_right'], listmeasure[i]['distance_from_2_points_left'], listmeasure[i]['distance_from_2_points_right'], listmeasure[i]['angle_left'], listmeasure[i]['angle_right']]
        features.append(feature)
        i=i+1
    return features

def get_features100(listmeasure):
    i=0
    features=[]
    t1=float(listmeasure[0]['time'])
    x_left=[]
    x_right=[]
    y_left=[]
    y_right=[]
    f=[float(listmeasure[0]['left_pupil_diameter']), float(listmeasure[0]['right_pupil_diameter']), int(listmeasure[0]['left_eye_blink']), int(listmeasure[0]['right_eye_blink']), int(listmeasure[0]['fixation_left']), int(listmeasure[0]['fixation_right']), int(listmeasure[0]['saccad_left']), int(listmeasure[0]['saccad_right']), float(listmeasure[0]['distance_from_2_points_left']), float(listmeasure[0]['distance_from_2_points_right']), listmeasure[0]['angle_left'], listmeasure[0]['angle_right'], [[0]], [[0]], [[0]], [[0]]]
    print(len(f))
    ts=[]
    cpt=0
    while(i<len(listmeasure)):
        time=float(listmeasure[i]['time'])-float(t1)
        if time<=100*(float(listmeasure[1]['time'])-(float(listmeasure[0]['time'])))//8:
            f[0]=f[0]+float(listmeasure[i]['left_pupil_diameter'])
            f[1]=f[1]+float(listmeasure[i]['right_pupil_diameter'])
            f[2]=f[2]+int(listmeasure[i]['left_eye_blink'])
            f[3]=f[3]+int(listmeasure[i]['right_eye_blink'])
            f[4]=f[4]+int(listmeasure[i]['fixation_left'])
            f[5]=f[5]+int(listmeasure[i]['fixation_right'])
            f[6]=f[6]+int(listmeasure[i]['saccad_left'])
            f[7]=f[7]+int(listmeasure[i]['saccad_right'])
            f[8]=f[8]+float(listmeasure[i]['distance_from_2_points_left'])
            f[9]=f[9]+float(listmeasure[i]['distance_from_2_points_right'])
            f[10]=f[10]+listmeasure[i]['angle_left']
            f[11]=f[11]+listmeasure[i]['angle_right']
            cpt=cpt+1
            if float(listmeasure[i]['gaze_left_x_y'].split(" ")[0])!=-1 :
                x_left.append(float(listmeasure[i]['gaze_left_x_y'].split(" ")[0]))
                y_left.append(float(listmeasure[i]['gaze_left_x_y'].split(" ")[1]))
            else :
                y_left.append(0)
                x_left.append(0)
            if float(listmeasure[i]['gaze_right_x_y'].split(" ")[0])!=-1 :
                x_right.append(float(listmeasure[i]['gaze_right_x_y'].split(" ")[0]))
                y_right.append(float(listmeasure[i]['gaze_right_x_y'].split(" ")[1]))
            else :
                x_right.append(0)
                y_right.append(0)
            ts.append(float(listmeasure[i]['time'])-float(listmeasure[0]['time']))
        else :
            traj_left_x, traj_left_y=trajectory(x_left, y_left, ts)
            traj_right_x, traj_right_y=trajectory(x_right, y_right, ts)
            listmeasure[i]['polyxleft']=traj_left_x
            listmeasure[i]['polyyleft']=traj_left_y
            listmeasure[i]['polyxright']=traj_right_x
            listmeasure[i]['polyyright']=traj_right_y
            f[0]=f[0]/cpt
            f[1]=f[1]/cpt
            features.append(f)
            f=[float(listmeasure[i]['left_pupil_diameter']), float(listmeasure[i]['right_pupil_diameter']), int(listmeasure[i]['left_eye_blink']), int(listmeasure[i]['right_eye_blink']), int(listmeasure[i]['fixation_left']), int(listmeasure[i]['fixation_right']),int(listmeasure[i]['saccad_left']), int(listmeasure[i]['saccad_right']), float(listmeasure[i]['distance_from_2_points_left']), float(listmeasure[i]['distance_from_2_points_right']), listmeasure[i]['angle_left'], listmeasure[i]['angle_right'], listmeasure[i]['polyxleft'],listmeasure[i]['polyyleft'],listmeasure[i]['polyxright'], listmeasure[i]['polyyright']]
            t1=listmeasure[i]['time']
            x_left=[]
            x_right=[]
            y_left=[]
            y_right=[]
            ts=[]
            i=i-1
            cpt=0
        i=i+1
    return features

#get all features
def calcul_features(measure):
    #measure statiques
    list_dict_features=validity(measure)
    
    #features mesures
    #blinks left eye right eye
    listmeasure=blink(list_dict_features)
    #fixation
    listmeasure=time_and_frequencies_fixations(listmeasure)
    listmeasure=distances(listmeasure)
    listmeasure=angles(listmeasure)
    features_1=get_features1(listmeasure)
    features_100=get_features100(listmeasure)
    print(len(features_100[0]))
    #listmeasure=trajectory(listmeasure)
    return features_100

measure=extract_measures("occulaire-s01e01.txt")
measure=calcul_features(measure)

file1=open("features-occulaire-s01e01.txt","w")
i=0
txt=""
while(i<len(measure)):
    j=0
    while(j<12):
        txt=txt+str(measure[i][j])+"\t"
        j=j+1
    while(j<len(measure[i])):
        k=0
        while(k<len(measure[i][j])):
            txt=txt+str(measure[i][j][k][0])+" "
            k=k+1
        txt=txt+"\t"
        j=j+1
    txt=txt+"\n"
    file1.write(txt)
    i=i+1
file1.close()

#polynome en fonction du temps : est-ce utile sachant que j'ai déjà la position en x et en y en fonction du temps
#serait-ce utile si j'avais une moyenne toutes les 100 ms.
#pour chaque position toutes les 8 ms, vecteur de changement d'état non étiqueté