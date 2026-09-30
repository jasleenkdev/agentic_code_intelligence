async function connect(id) {
  await this.prepare();
  return id;
}

async function prepare() {
  return true;
}

function listIds() {
  return [1, 2].map((x) => x * 2);
}

module.exports = { connect, prepare, listIds };
