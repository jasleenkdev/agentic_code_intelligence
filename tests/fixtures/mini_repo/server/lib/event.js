const EventManager = function EventManager() {};

EventManager.prototype.emit = function emit(event, payload) {
  return [event, payload];
};

module.exports = EventManager;
