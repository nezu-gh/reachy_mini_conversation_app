#!/usr/bin/env bash
# patch-daemon-camera.sh — Fix broken v4l2h264enc on Reachy Mini wireless (RPi 5)
#
# The daemon's WebRTC pipeline uses v4l2h264enc (hardware H264 encoder) to
# stream camera video. On some RPi 5 units this encoder fails with "not enough
# memory or failing driver", which stalls the entire GStreamer tee and prevents
# both the unixfdsink (camera socket) and webrtcsink (WebRTC video) from
# receiving any frames.
#
# This script patches the daemon to auto-detect the broken encoder at startup
# and fall back to openh264enc (software encoder). If v4l2h264enc works, it
# is used as before.
#
# Supports:
#   - v1.5.x: patches webrtc_daemon.py (_configure_video)
#   - v1.6.x: patches media_server.py  (_build_rpi_encoder_branch)
#
# After patching, the daemon process must be fully restarted (kill + systemd
# auto-restart), not just the API restart — the GStreamer pipeline is created
# once at daemon init.
#
# Usage:
#   ssh pollen@192.168.178.127 'bash -s' < scripts/patch-daemon-camera.sh
#   # or copy to robot and run locally

set -euo pipefail

SITE_PACKAGES="/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/media"
FILE_V16="${SITE_PACKAGES}/media_server.py"
FILE_V15="${SITE_PACKAGES}/webrtc_daemon.py"

# Determine which version we're dealing with
if [ -f "$FILE_V16" ]; then
    TARGET="$FILE_V16"
    VERSION="v1.6+"
elif [ -f "$FILE_V15" ]; then
    TARGET="$FILE_V15"
    VERSION="v1.5"
else
    echo "ERROR: Neither media_server.py nor webrtc_daemon.py found in $SITE_PACKAGES"
    exit 1
fi

echo "Detected reachy-mini $VERSION — patching $TARGET"

# Check if already patched
if grep -q 'openh264enc' "$TARGET"; then
    echo "Already patched — openh264enc fallback is present"
    exit 0
fi

# Check openh264enc is available
if ! gst-inspect-1.0 openh264enc >/dev/null 2>&1; then
    echo "ERROR: openh264enc GStreamer plugin not found"
    echo "Install with: sudo apt-get install gstreamer1.0-plugins-bad"
    exit 1
fi

# Backup
cp "$TARGET" "${TARGET}.bak"
echo "Backup saved to ${TARGET}.bak"

# Apply version-specific patch
python3 - "$TARGET" "$VERSION" << 'PYEOF'
import sys

filepath = sys.argv[1]
version = sys.argv[2]

with open(filepath, "r") as f:
    content = f.read()

if version == "v1.6+":
    # v1.6+: _build_rpi_encoder_branch in media_server.py
    old = '''    def _build_rpi_encoder_branch(
        self,
        queue_webrtc: Gst.Element,
        pipeline: Gst.Pipeline,
        webrtcsink: Gst.Element,
    ) -> None:
        """Build the RPi hardware H264 encoder branch.

        webrtcsink does not have v4l2h264enc, so we encode explicitly on RPi.
        """
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)

        # H264 Level 3.1 + Constrained Baseline for Safari/WebKit compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        if not all([v4l2h264enc, capsfilter_h264]):
            raise RuntimeError("Failed to create RPi H264 encoder elements")

        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)

        queue_webrtc.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)'''

    new = '''    def _build_rpi_encoder_branch(
        self,
        queue_webrtc: Gst.Element,
        pipeline: Gst.Pipeline,
        webrtcsink: Gst.Element,
    ) -> None:
        """Build the RPi H264 encoder branch.

        Uses openh264enc (software) as fallback when v4l2h264enc (hardware)
        fails — which happens on some RPi 5 units with "not enough memory
        or failing driver".
        """
        # Try hardware encoder first, fall back to software
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        use_software = v4l2h264enc is None

        if not use_software:
            # Quick probe: try setting state to READY to detect driver issues
            test_pipe = Gst.Pipeline.new("encoder_test")
            test_src = Gst.ElementFactory.make("videotestsrc")
            test_src.set_property("num-buffers", 1)
            test_caps = Gst.ElementFactory.make("capsfilter")
            test_caps.set_property("caps", Gst.Caps.from_string(
                "video/x-raw,width=320,height=240,format=I420,framerate=1/1"
            ))
            test_enc = Gst.ElementFactory.make("v4l2h264enc")
            test_sink = Gst.ElementFactory.make("fakesink")
            for e in [test_src, test_caps, test_enc, test_sink]:
                test_pipe.add(e)
            test_src.link(test_caps)
            test_caps.link(test_enc)
            test_enc.link(test_sink)
            ret = test_pipe.set_state(Gst.State.PAUSED)
            if ret == Gst.StateChangeReturn.ASYNC:
                ret2 = test_pipe.get_state(2 * Gst.SECOND)
                if ret2[0] == Gst.StateChangeReturn.FAILURE:
                    use_software = True
            elif ret == Gst.StateChangeReturn.FAILURE:
                use_software = True
            test_pipe.set_state(Gst.State.NULL)
            del test_pipe

        if use_software:
            self._logger.warning(
                "v4l2h264enc unavailable or broken, falling back to openh264enc"
            )
            videoconvert_enc = Gst.ElementFactory.make("videoconvert", "videoconvert_enc")
            encoder = Gst.ElementFactory.make("openh264enc")
            if encoder is None:
                raise RuntimeError(
                    "Neither v4l2h264enc nor openh264enc available. "
                    "Install gstreamer1.0-plugins-bad for openh264enc."
                )
            encoder.set_property("bitrate", 2000000)
            encoder.set_property("complexity", 0)
            caps_h264 = Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au,"
                "profile=(string)constrained-baseline"
            )
            capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
            capsfilter_h264.set_property("caps", caps_h264)

            pipeline.add(videoconvert_enc)
            pipeline.add(encoder)
            pipeline.add(capsfilter_h264)

            queue_webrtc.link(videoconvert_enc)
            videoconvert_enc.link(encoder)
            encoder.link(capsfilter_h264)
            capsfilter_h264.link(webrtcsink)
        else:
            extra_controls_structure = Gst.Structure.new_empty("extra-controls")
            extra_controls_structure.set_value("repeat_sequence_header", 1)
            extra_controls_structure.set_value("video_bitrate", 5_000_000)
            extra_controls_structure.set_value("h264_i_frame_period", 60)
            extra_controls_structure.set_value("video_gop_size", 256)
            v4l2h264enc.set_property("extra-controls", extra_controls_structure)

            caps_h264 = Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au,"
                "level=(string)3.1,profile=(string)constrained-baseline"
            )
            capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
            capsfilter_h264.set_property("caps", caps_h264)

            if not all([v4l2h264enc, capsfilter_h264]):
                raise RuntimeError("Failed to create RPi H264 encoder elements")

            pipeline.add(v4l2h264enc)
            pipeline.add(capsfilter_h264)

            queue_webrtc.link(v4l2h264enc)
            v4l2h264enc.link(capsfilter_h264)
            capsfilter_h264.link(webrtcsink)'''

elif version == "v1.5":
    # v1.5: _configure_video in webrtc_daemon.py
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

else:
    print(f"ERROR: Unknown version {version}", file=sys.stderr)
    sys.exit(1)

if old not in content:
    print(f"ERROR: Could not find expected code block in {filepath}", file=sys.stderr)
    print("The file may have been modified or is a different sub-version.", file=sys.stderr)
    sys.exit(1)

content = content.replace(old, new)
with open(filepath, "w") as f:
    f.write(content)
print("Patch applied successfully")
PYEOF

echo ""
echo "Patch applied. Now restart the daemon process:"
echo "  kill \$(pgrep -f 'reachy_mini.daemon.app.main')"
echo "  # systemd will auto-restart it"
