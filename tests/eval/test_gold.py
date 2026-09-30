from eval.generate_gold import _doc_contradicts_path

SERVICES = {"tuya", "melcloud", "bluetooth", "mqtt", "nuki", "lan-manager"}


def test_copy_pasted_doc_is_detected():
    assert _doc_contradicts_path(
        "Poll values of an Tuya device", "server/services/melcloud/lib/melcloud.poll.js", SERVICES
    )


def test_legitimate_mentions_are_kept():
    assert not _doc_contradicts_path(
        "Poll values of an Tuya device", "server/services/tuya/lib/tuya.poll.js", SERVICES
    )
    assert not _doc_contradicts_path(
        "Discover Nuki devices through MQTT", "server/services/nuki/lib/mqtt/x.js", SERVICES
    )
    assert not _doc_contradicts_path("Tuya device helper", "server/lib/device/x.js", SERVICES)
