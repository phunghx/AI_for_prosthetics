for i in {0..19}
do
   python server.py -l 127.0.0.1 -p 500"$i" &
done
