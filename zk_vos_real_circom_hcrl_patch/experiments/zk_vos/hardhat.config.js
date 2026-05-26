require("@nomiclabs/hardhat-ethers");
require("hardhat-gas-reporter");

module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: false,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
      blockGasLimit: 60000000,
    },
  },
  gasReporter: {
    enabled: true,
    currency: "USD",
    showTimeSpent: true,
    excludeContracts: [],
  },
};
