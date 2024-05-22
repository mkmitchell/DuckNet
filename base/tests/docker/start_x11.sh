#!/bin/bash

export DISPLAY=${DISPLAY:-:99} 
xdpyinfo
if which x11vnc &>/dev/null; then
  ! pgrep -a x11vnc && x11vnc -bg -forever -nopw -quiet -display WAIT$DISPLAY &
fi
! pgrep -a Xvfb && Xvfb $DISPLAY -screen 0 1280x1024x16 &
sleep 0.25
echo "IP: $(hostname -I) ($(hostname))"

openbox &> /dev/null &


if [ "$#" != 0 ]; then
	$@
fi

/bin/bash

