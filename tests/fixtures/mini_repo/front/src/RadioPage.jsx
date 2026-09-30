import { Link } from 'preact-router/match';
import { formatName } from './utils';

export default class RadioPage {
  render() {
    const label = formatName('radio');
    return (
      <div class="page">
        <Link href="/dashboard/integration/device/radio">{label}</Link>
      </div>
    );
  }
}
