from datetime import timezone
import datetime as dt
import time
import pytz

now = time.time()

now_dt_ny = dt.datetime.now(tz=timezone(offset=dt.timedelta(hours=-5)))
now_dt_berlin = dt.datetime.now(tz=timezone(offset=dt.timedelta(hours=2)))
now_nyc_tstamp = now_dt_ny.timestamp()
now_berlin_tstamp = now_dt_berlin.timestamp()

now_nyc_2_berlin = dt.datetime.fromtimestamp(now_nyc_tstamp, tz=timezone(offset=dt.timedelta(hours=0)))


dt_ny = dt.datetime(year=2018,month=6,day=25,hour=15,minute=21,second=33,microsecond=0, tzinfo=timezone(offset=dt.timedelta(hours=-5)))
dt_berlin = dt.datetime(year=2018,month=6,day=25,hour=15,minute=21,second=33,microsecond=0, tzinfo=timezone(offset=dt.timedelta(hours=2)))
ny_tstamp = dt_ny.timestamp()
berlin_tstamp = dt_berlin.timestamp()


time1 = dt.datetime(year=2018,month=6,day=25,hour=15,minute=21,second=33,microsecond=0, tzinfo=timezone.utc)
tst = time1.timestamp()
