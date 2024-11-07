<p>
cd lidar_camera/ <br />
pip install -r requirements.txt <br />
git clone https://github.com/Slamtec/rplidar_sdk.git <br />
./build.sh && python3 setup.py build_ext --inplace -j4 <br />
python main.py <br />
При ошибке: "Error, unexpected error, code: 80008004", выполнить "sudo chmod 777 /dev/ttyUSB0" <br />
</p>
