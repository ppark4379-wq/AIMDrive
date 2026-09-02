#!/bin/bash
protoc \
	--plugin=protoc-gen-doc=./protoc-gen-doc.exe \
	--doc_out=. \
  	--doc_opt=markdown,docs.md \
	./morai/common/*.proto \
	./morai/actor/*.proto \
	./morai/environment/*.proto \
	./morai/infrastructure/*.proto \
	./morai/map/*.proto \
	./morai/scenario/*.proto \
	./morai/sensor/*.proto \
	./morai/simulation/*.proto \
	./morai/simulator/*.proto \
	./morai/util/*.proto
