const { EVENTS } = require('../constants');

/**
 * @description Scan for radio peripherals.
 */
async function scan(timeout) {
  await this.prepare();
  if (timeout > 10) {
    this.eventManager.emit(EVENTS.RADIO.SCAN_STARTED, 'radio://settings/bluetooth');
  }
  const ids = await this.listIds();
  for (const id of ids) {
    await this.connect(id);
  }
  return ids;
}

module.exports = { scan };
