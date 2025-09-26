// data/dukascopy_client.cjs
const { getHistoricalRates } = require("dukascopy-node");
const [symbol, timeframe, from, to] = process.argv.slice(2);

(async () => {
  try {
    const data = await getHistoricalRates({
      instrument: symbol.toLowerCase(),
      dates: {
        from: new Date(from),
        to: new Date(to)
      },
      timeframe: timeframe.toLowerCase(),
      format: "json",
      volumes: true,
      ignoreFlats: true
    });
    console.log(JSON.stringify(data));
  } catch (error) {
    console.error("❌ Error:", error.message);
    process.exit(1);
  }
})();
