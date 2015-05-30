##EDIT THE FOLLOWING LINES ACCORDING TO YOUR SYSTEM CONFIGURATION
CC = g++
CXX = g++

## where the binary directory will be created to store executables after make install
PROJECT=.

## include paths for fft3, magick++ and gsl
FFTW3_IPATH = -I/usr/local/include
IMAGICK_IPATH = -I/usr/local/include
GSL_IPATH = -I/usr/local/include/gsl

## library paths for fft3, magick++ and gsl
FFTW3_LPATH = -L/usr/local/lib
IMAGICK_LPATH = -L/usr/local/lib
GSL_LPATH = -L/usr/local/lib


## In principle, no change should be needed beyond this line
ARRAY_IPATH = -I$(PROJECT)/arrayClasses 
IPATH = $(ARRAY_IPATH) $(GSL_IPATH) $(FFTW3_IPATH) $(JAMA_IPATH) $(IMAGICK_IPATH)   
LPATH = $(GSL_LPATH) $(FFTW3_LPATH) $(IMAGICK_LPATH)   
CFLAGS =  -O3 -ffast-math -Wall 
#-arch x86_64 
#COMPILE.cpp = $(CC) $(IPATH)  $(CFLAGS) -c
COMPILE.cpp = $(CC) $(ARRAY_IPATH) `pkg-config --cflags fftw3 GraphicsMagick++ gsl` $(CFLAGS) -c
LIBPATH = -lfftw3 -lMagick++ -lstdc++ -lgsl -lgslcblas
LDLIBS = `pkg-config --libs fftw3 GraphicsMagick++ gsl` 
COMPILE.o = $(CC)   
#COMPILE.o = $(CC) $(IPATH)  $(CFLAGS) 

DIR.h = \
	arrayClasses \

SOURCES.h = \
	affineReg.h \
	arrayClasses/byteSwap.h \
	deformableObject.h \
	deformableImage.h \
	diffeomorphism.h \
	greedyImage.h \
	ImageEvolution.h \
	ImageMatching.h \
	ImageMatchingAffine.h \
	arrayClasses/Ivector.h \
	kernels.h \
	landmarkMatching.h \
	matchingBase.h \
	arrayClasses/matrix.h \
	morphingNew.h \
	optim.h \
	optimF.h\
	pointSet.h \
	pointSetEvolution.h \
	pointSetFunctors.h \
	pointSetMatching.h \
	PointSetMatchingAffine.h \
	param_matching.h \
	shooting.h \
	tensorField.h \
	arrayClasses/VectorBase.h\
	Vector.h \
	velocity.h


SOURCES.cpp = \
	deformableObjects.cpp \
	landmarkMatching.cpp \
	param_matching.cpp \
	pointSet.cpp \
	PointSetVectorTransforms.cpp \
	shooting.cpp \
	shootingSEG.cpp \
	shootingSym.cpp \
	velocity.cpp 

MAINS.cpp = \
	affine_registration.cpp \
	greedyImageMatching.cpp \
	initialMomentumLandmarkMatching.cpp \
	initialMomentumMatching.cpp \
	initialMomentumMatchingSYM.cpp \
	initialMomentumSegmentation.cpp \
	metamorphosis.cpp \
	momentumAverage.cpp \
	momentumCovMatrix.cpp \
	parallelTranslation.cpp \
	printDefaultParam.cpp \
	procrusteanAverage.cpp \
	shoot.cpp \
	tensorCurvature.cpp \
	velocityMatching.cpp 


MAINS = $(MAINS.cpp:%.cpp=%)
SOURCES.o = $(SOURCES.cpp:%.cpp=%.o)

#%.d: %.cpp
#	$(SHELL) -ec '$(CC) -M  $(IPATH)  $(CFLAGS) $< | sed '\"s/$*.o/& $@/g'\" > $@'
%.d: %.cpp $(SOURCES.h)
	$(SHELL) -ec '$(CC) -M  $(ARRAY_IPATH) `pkg-config --cflags fftw3 GraphicsMagick++ gsl`  $(CFLAGS) $< > $@'

-include $(SOURCES.cpp:.cpp=.d)
-include $(MAINS.cpp:.cpp=.d)


%.o: %.cpp 
	$(COMPILE.cpp) $<
%: %.o $(SOURCES.o)
	$(COMPILE.o) $< -o $@ $(SOURCES.o) $(LDLIBS)

all: metamorphosis affine_registration initialMomentumMatching initialMomentumMatchingSYM initialMomentumSegmentation velocityMatching parallelTranslation procrusteanAverage momentumAverage shoot momentumCovMatrix initialMomentumLandmarkMatching printDefaultParam


preinstall:
	mkdir -p $(PROJECT)/include  
	cp -R ./arrayClasses $(PROJECT)/include

doc:
	doxygen matching.dox

clean:
	rm -f *.o *.d

tar: $(SOURCES.cpp) $(SOURCES.h)
	tar cvzf matching.tgz $(DIR.h) $(SOURCES.cpp) $(SOURCES.h) $(MAINS.cpp) Makefile.template matching.dox README.pdf

