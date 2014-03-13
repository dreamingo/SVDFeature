.PHONY: all clean

all:
	cd src/; make; cd - 

clean:
	cd src/; make clean; cd -
	cd model/; rm *.model; cd -

