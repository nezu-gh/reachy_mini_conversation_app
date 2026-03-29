#!/usr/bin/env bash
# patch-daemon-camera.sh — Fix broken v4l2h264enc on Reachy Mini wireless (RPi 5)
#
# The daemon's WebRTC pipeline uses v4l2h264enc (hardware H264 encoder) to
# stream camera video. On some RPi 5 units this encoder fails with "not enough
# memory or failing driver", which stalls the entire GStreamer tee and prevents
# both the unixfdsink (camera socket) and webrtcsink (WebRTC video) from
# receiving any frames.
#
# This script patches the daemon to use openh264enc (software encoder) instead.
# After patching, the daemon process must be fully restarted (kill + systemd
# restart), not just the API restart — the GstWebRTC object is created once at
# daemon init.
#
# Usage:
#   ssh pollen@192.168.178.127 'bash -s' < scripts/patch-daemon-camera.sh
#   # or copy to robot and run locally

set -euo pipefail

DAEMON_FILE="/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/media/webrtc_daemon.py"
BACKUP="${DAEMON_FILE}.bak"

if [ ! -f "$DAEMON_FILE" ]; then
    echo "ERROR: Daemon file not found: $DAEMON_FILE"
    exit 1
fi

# Check if already patched
if grep -q 'openh264enc' "$DAEMON_FILE"; then
    echo "Already patched — openh264enc is present in $DAEMON_FILE"
    exit 0
fi

# Check openh264enc is available
if ! gst-inspect-1.0 openh264enc >/dev/null 2>&1; then
    echo "ERROR: openh264enc GStreamer plugin not found"
    echo "Install with: sudo apt-get install gstreamer1.0-plugins-bad"
    exit 1
fi

# Backup
cp "$DAEMON_FILE" "$BACKUP"
echo "Backup saved to $BACKUP"

# Apply patch using Python for reliable multi-line replacement
python3 << 'PYEOF'
import sys

filepath = "/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/media/webrtc_daemon.py"
with open(filepath, "r") as f:
    content = f.read()

old = '''        queue_encoder = Gst.ElementFactory.make("queue", "queue_encoder")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        # doc: https://docs.qualcomm.com/doc/80-70014-50/topic/v4l2h264enc.html
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        # Use H264 Level 3.1 + Constrained Baseline for Safari/WebKit compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        if not all(
            [
                camerasrc,
                capsfilter,
                tee,
                queue_unixfd,
                unixfdsink,
                queue_encoder,
                v4l2h264enc,
                capsfilter_h264,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        pipeline.add(camerasrc)
        pipeline.add(capsfilter)
        pipeline.add(tee)
        pipeline.add(queue_unixfd)
        pipeline.add(unixfdsink)
        pipeline.add(queue_encoder)
        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)

        camerasrc.link(capsfilter)
        capsfilter.link(tee)
        tee.link(queue_unixfd)
        queue_unixfd.link(unixfdsink)
        tee.link(queue_encoder)
        queue_encoder.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)'''

new = '''        queue_encoder = Gst.ElementFactory.make("queue", "queue_encoder")
        # videoconvert needed: openh264enc expects I420, camera outputs YUY2
        videoconvert_enc = Gst.ElementFactory.make("videoconvert", "videoconvert_enc")
        # Use openh264enc as software fallback (v4l2h264enc broken on this RPi)
        openh264enc = Gst.ElementFactory.make("openh264enc")
        openh264enc.set_property("bitrate", 2000000)  # 2 Mbps
        openh264enc.set_property("complexity", 0)  # low complexity for RPi
        # H264 caps for WebRTC compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        if not all(
            [
                camerasrc,
                capsfilter,
                tee,
                queue_unixfd,
                unixfdsink,
                queue_encoder,
                videoconvert_enc,
                openh264enc,
                capsfilter_h264,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        pipeline.add(camerasrc)
        pipeline.add(capsfilter)
        pipeline.add(tee)
        pipeline.add(queue_unixfd)
        pipeline.add(unixfdsink)
        pipeline.add(queue_encoder)
        pipeline.add(videoconvert_enc)
        pipeline.add(openh264enc)
        pipeline.add(capsfilter_h264)

        camerasrc.link(capsfilter)
        capsfilter.link(tee)
        tee.link(queue_unixfd)
        queue_unixfd.link(unixfdsink)
        tee.link(queue_encoder)
        queue_encoder.link(videoconvert_enc)
        videoconvert_enc.link(openh264enc)
        openh264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)'''

if old not in content:
    print("ERROR: Could not find v4l2h264enc code block to replace", file=sys.stderr)
    sys.exit(1)

content = content.replace(old, new)
with open(filepath, "w") as f:
    f.write(content)
print("Patch applied successfully")
PYEOF

echo ""
echo "Patch applied. Now restart the daemon process:"
echo "  kill \$(pgrep -f 'reachy_mini.daemon.app.main')"
echo "  # systemd will auto-restart it, or manually:"
echo "  # source /venvs/mini_daemon/bin/activate"
echo "  # GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu/ python -u -m reachy_mini.daemon.app.main --wireless-version --no-wake-up-on-start"
