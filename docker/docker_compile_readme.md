VERSION=1.0
docker build -t hexinzi/npp:${VERSION} .
docker push hexinzi/npp:${VERSION}
