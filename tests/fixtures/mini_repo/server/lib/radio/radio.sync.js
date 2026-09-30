const { EVENTS } = require('../constants');

async function sync(ids) {
  const config = await this.loadConfig();
  const status = await this.getStatus();
  for (const id of ids) {
    const settings = await this.getSettings('global');
    this.eventManager.emit(EVENTS.RADIO.SCAN_STARTED, id, settings, config, status);
  }
  return 'radio://settings/bluetooth';
}

function unusedHelper() {
  return 1;
}

module.exports = { sync };
