<template>
  <div id="app">
    <el-row margin-top="3px" type="flex" align="middle">
        <font color="#000000" size="4" font-family="Helvetica"><b>&nbsp;&nbsp; Dota2-Your Next Handy Weapon</b></font>
    </el-row>
    <br>
    <el-row :gutter="30" style="margin-left:10px;margin-right:10px">
      <el-col :span="4">
        Your Dota2 UID: 
        <br><br>
        <el-input v-model="uidText" placeholder="UID Here"></el-input>
        <br><br>
          <el-button type="primary" @click="getResult">Recommend</el-button>
      </el-col>
      <el-col :span="10">
        <font color="#000000" size="4" font-family="Helvetica"><b>Your Most Played Heroes</b></font>
        <br><br>
        <el-table
            :data="result.preference"
            style="width: 100%">
                <el-table-column
                    prop="localized_name"
                    label="Name"
                    width="180">
                </el-table-column>
                <el-table-column
                    prop="games"
                    label="Games"
                    width="100">
                </el-table-column>
                <el-table-column
                    prop="win"
                    label="Wins"
                    width="100">
                </el-table-column>
                <el-table-column
                    prop="win_rate"
                    label="Win Rate"
                    width="100">
                </el-table-column>
                <el-table-column
                    prop="preference_factor"
                    label="Preference"
                    width="100">
                </el-table-column>
            </el-table>
        <br><br>
      </el-col>
      <el-col :span="10">
        <font color="#000000" size="4" font-family="Helvetica"><b>Similar Top Players</b></font>
        <br><br>
        <el-table
            :data="result.similar_players"
            style="width: 100%">
                <el-table-column
                    prop="id"
                    label="UID"
                    width="200">
                </el-table-column>
                <el-table-column
                    prop="name"
                    label="User Name"
                    width="400">
                </el-table-column>
            </el-table>
        <br>
        <font color="#000000" size="4" font-family="Helvetica"><b>Recommended Heroes</b></font>
        <br><br>
        <el-row :gutter="20">
        <div
          v-for="h in result.recommended_heroes"
          :key="h.id"
        >  
          <el-tag :span="3" class="mx-1">
            {{ h.localized_name }}
          </el-tag>
          &nbsp;&nbsp;
         </div>
        </el-row>
      </el-col>

    </el-row>
  </div>
</template>

<script>

import axios from "axios"

export default {
  name: 'App',
  data() {
    return {
      uidText: '',
      result: {
        "preference": [],
        "recommended_heroes": [],
        "similar_players": []
      }

    }
  },
  methods: {
    getResult() {
      axios.get('http://127.0.0.1:5000/get_result?uid=' + this.uidText + '&top_p=10&top_r=6&top_s=5').then((response) => {
        this.result = response.data;
      });
      return 0;
    }
  },
  components: { }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 10px;
}
</style>