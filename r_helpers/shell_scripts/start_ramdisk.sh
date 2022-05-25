#/bin/sh
mkdir -p "$1" &&
sudo chmod 777 "$1" #&&
rm -rf "$1"/* #&&
sudo mount -t tmpfs -o size=$2 ramdisk "$1" &&
echo "ramdisk of size" $2" successfully started at" \"$1\"
