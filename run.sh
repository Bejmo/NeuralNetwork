#!/bin/bash
# run.sh

SRC_DIR="src/main/java/"
BIN_DIR="bin"
MAIN_CLASS="Main"


echo "Adding some modules"
module add java/openjdk-17


echo "#################"
echo "     COMPILING     "
echo "#################"

# Create (if doesn't exist) the bin directory
mkdir -p "$BIN_DIR"
# Compile all files
javac -d "$BIN_DIR" $(find "$SRC_DIR" -name "*.java")
# Verify compilation
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful. .class files are in $BIN_DIR"


echo "#################"
echo "      RUNNING      "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network
nice -n 19 java -cp "$BIN_DIR" "$MAIN_CLASS"

# Verify execution
if [ $? -ne 0 ]; then
    echo "ERROR: Execution failed!"
    exit 1
fi