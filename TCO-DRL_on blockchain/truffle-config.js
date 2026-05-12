module.exports = {
  contracts_directory: "./contracts",
  contracts_build_directory: "./build/contracts",
  migrations_directory: "./migrations",

  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*",
      websockets: true
    }
  },

  compilers: {
    solc: {
      version: "0.5.12",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }
  }
};