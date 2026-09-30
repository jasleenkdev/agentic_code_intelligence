const { scan } = require('./radio.scan');
const { connect, prepare, listIds } = require('./radio.connect');

const RadioManager = function RadioManager(eventManager) {
  this.eventManager = eventManager;
};

RadioManager.prototype.scan = scan;
RadioManager.prototype.connect = connect;
RadioManager.prototype.prepare = prepare;
RadioManager.prototype.listIds = listIds;

module.exports = RadioManager;
