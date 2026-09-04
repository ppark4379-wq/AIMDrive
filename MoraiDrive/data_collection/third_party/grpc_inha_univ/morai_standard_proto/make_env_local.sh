#!/bin/bash 

echo Start 

# plugin path must be an absolute path :(
#GRPC_PLUGIN_PATH="/home/morai/.local/bin"
GRPC_CSHARP_PLUGIN_PATH="C:/Users/hello/.nuget/packages/grpc.tools/2.38.1/tools/windows_x64/grpc_csharp_plugin.exe"
PROTO_FILES_DIRS=(
"/common/"
"/actor/"
"/environment/"
"/infrastructure/"
"/map/"
"/scenario/"
"/sensor/"
"/simulation/"
"/simulator/"
"/util/"
)

# C++
# echo c++
# PROTOC_OUTPUT_DIR="./output_cpp"

# for PROTO_DIR in ${PROTO_FILES_DIRS[@]}
# do
# 	#OUTPUT_DIR=$PROTOC_OUTPUT_DIR$PROTO_DIR
# 	OUTPUT_DIR=$PROTOC_OUTPUT_DIR
# 	PROTO_PATH=./morai$PROTO_DIR*.proto
	
# 	mkdir -p $OUTPUT_DIR
# 	protoc -I . --cpp_out=$OUTPUT_DIR $PROTO_PATH
# 	protoc -I . --grpc_out=$OUTPUT_DIR --plugin=protoc-gen-grpc=$GRPC_PLUGIN_PATH/grpc_cpp_plugin $PROTO_PATH
# done

# C#
echo c#
PROTOC_OUTPUT_DIR="./output_cs"
for PROTO_DIR in ${PROTO_FILES_DIRS[@]}
do
	OUTPUT_DIR=$PROTOC_OUTPUT_DIR$PROTO_DIR
	#OUTPUT_DIR=$PROTOC_OUTPUT_DIR
	PROTO_PATH=./morai$PROTO_DIR*.proto
	
	mkdir -p $OUTPUT_DIR
	protoc -I . --csharp_out=$OUTPUT_DIR $PROTO_PATH
	protoc -I . --grpc_out=$OUTPUT_DIR --plugin=protoc-gen-grpc=$GRPC_CSHARP_PLUGIN_PATH $PROTO_PATH
done

# python
# echo python
# PROTOC_OUTPUT_DIR="./output_py"
# for PROTO_DIR in ${PROTO_FILES_DIRS[@]}
# do
# 	#OUTPUT_DIR=$PROTOC_OUTPUT_DIR$PROTO_DIR
# 	OUTPUT_DIR=$PROTOC_OUTPUT_DIR
# 	PROTO_PATH=./morai$PROTO_DIR*.proto
	
# 	mkdir -p $OUTPUT_DIR
# 	protoc -I . --python_out=$OUTPUT_DIR $PROTO_PATH
# 	protoc -I . --grpc_out=$OUTPUT_DIR --plugin=protoc-gen-grpc=$GRPC_PLUGIN_PATH/grpc_python_plugin $PROTO_PATH
# done

echo Finished