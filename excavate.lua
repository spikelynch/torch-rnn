require 'torch'
require 'nn'

local cjson = require 'cjson'
local cjson2 = cjson.new()
local pretty = require "resty.prettycjson"

utf8 = require 'lua-utf8'

require 'LanguageModel'

local GARBAGE_INTERVAL = 1000


-- TODO: 

-- if called without vocab + suppress, do simple oulipo
-- apply alliteration as a filter to vocab

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'Projects/Musketeers/Musketeers2/Musketeers2_cp_176000.t7')
cmd:option('-vocab', '')
cmd:option('-notpunct', '’')
--cmd:option('-notpunct', '')
cmd:option('-alliterate', '')
cmd:option('-suppress', '')
cmd:option('-excavate', 0)
cmd:option('-length', 1000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', .3)
cmd:option('-name', 'excavate')
cmd:option('-outdir', '/Users/mike/Desktop/NaNoGenMo2019/Samples/')

local END_OFFSET = 5

local opt = cmd:parse(arg)


--print("Start", collectgarbage('count') * 1024)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local tokens = {}
local punctuation = {}

local vocab_filter = nil

model:evaluate()

--print("Loaded RNN", collectgarbage('count') * 1024)


local punctpat = nil
if opt.notpunct ~= '' then
  punctpat = '[' .. opt.notpunct .. ']'
end

for idx, token in pairs(model.idx_to_token) do
  tokens[idx] = 1
  if token:match('%W') then
    if not punctpat or not (utf8.match(token, punctpat)) then 
      punctuation[idx] = 1
    end
  end
end



function utf8first(s)
  local o2 = utf8.offset(s, 1)
  if o2 == nil then
    return s
  else
    return s:sub(1, o2 - 1)
  end
end


function get_matches(ws)
  local matches = {}
  for p, _ in pairs(punctuation) do
    matches[p] = 1
  end
  if ws then
    for _, w in pairs(ws) do
      if w then
        local idx = model.token_to_idx[utf8first(w)]
        if idx ~= nil then
          matches[idx] = 1
        else
          -- print("unknown first char in '" .. w .. "'")
          -- print("'" .. utf8first(w) .. "'")
        end
      end
    end
  end
  return matches
end




function matches_to_weights(matches)
  local weights = {}
  for idx, _ in pairs(tokens) do
    if matches[idx] ~= nil then
      weights[idx] = 1
    else
      weights[idx] = 0
    end
  end
  return weights
end


-- input to this is a list of [ index, words ]
-- output is just words

function init_vocab_orig(ws)
  local v = {}
  for i, iw in pairs(ws) do
    v[i] = iw[2]
  end
  return v
end


-- lookahead = { unpack(words, index, index + MAX_AHEAD - 1) }

function init_vocab_orig_2(ws, start, nwords)
  local v = {}
  for i = start, start + nwords - 1 do 
    v[i - start + 1] = ws[i][2]
  end
  return v
end


function init_vocab(ws)
  local v = {}
  for i = 1, #ws do 
    v[i] = ws[i][2]
  end
  return v
end


function prune_vocab(ov, next_char)
  local v = { "\n", " " }
  for _, w in pairs(ov) do
    local f = utf8first(w)
    if next_char == f then
      local n = w:sub(#f + 1, #w)
      if n then
        v[#v+1] = n
      end
    end
  end
  return v
end


function make_vocab(vocab_gen)
  return coroutine.create(function(prev_char)
    local ok, vocab = coroutine.resume(vocab_gen, nil)
    local current_word = ''
    while ok
      do
        local weights = {}
        local matches = get_matches(vocab)
        local weights = matches_to_weights(matches)
        p = coroutine.yield(weights)
        local next_idx = p[{1,1}]
        local next_char = model.idx_to_token[next_idx]
        if punctuation[next_idx] then
          -- print('"' .. current_word .. '" -> "' .. next_char .. '"')
          ok, vocab = coroutine.resume(vocab_gen, current_word)
          current_word = ''
        else
          current_word = current_word .. next_char
          vocab = prune_vocab(vocab, next_char)
          if #vocab < 1 then
            if coroutine.status(vocab_gen) == 'dead' then
              ok = false
            else
              ok, vocab = coroutine.resume(vocab_gen, nil)
            end
          end
        end
      end
      print("Vocabulary finished")
  end)
end


local fetch_word = coroutine.create(function()
  --print("Opening " .. opt.vocab)
  local f = io.open(opt.vocab, 'r')
  if not f then
    print("couldn't open " .. opt.vocab)
    os.exit()
  end
  local line
  repeat
    line = f:read()
    if line then
      local i, w = line:match("([^,]+),(.*)")
      if not vocab_filter or vocab_filter(w) then
        coroutine.yield({ tonumber(i), w })
      end
    end
  until line == nil 
end)


-- this depends on having the whole vocab loaded into memory

local basic_vocab = coroutine.create(function(uw)
  local words = {}
  repeat
    vocab_left, words[#words + 1] = coroutine.resume(fetch_word)
  until coroutine.status(fetch_word) == 'dead'
  local vocab = init_vocab(words)
  while true do
    coroutine.yield(vocab)
  end
end)


local MAX_AHEAD = 500

local word_indices = {}



-- first stab at pulling words off the vocab one at a time

local excavate_vocab = coroutine.create(function(uw)
  local used_word = ''
  local words = {}
  local vocab_left = true
  local tick = 0
  for j = 1, MAX_AHEAD do
    vocab_left, words[j] = coroutine.resume(fetch_word)
  end
  repeat
    local index = 1
    while index <= #words and not utf8.match(words[index][2], '^' .. used_word) do
        index = index + 1
    end
    if #used_word > 0 then
      word_indices[#word_indices + 1] = { words[index][1], used_word }
    end
    if index < #words then
      for j = index + 1, #words do
        words[j - index] = words[j]
      end
    end
    for j = #words - index + 1, #words do
      if coroutine.status(fetch_word) ~= 'dead' then
        vocab_left, words[j] = coroutine.resume(fetch_word)
      end
    end
    local vocab = init_vocab(words)
    used_word = coroutine.yield(vocab)
    tick = tick + 1
    if tick % GARBAGE_INTERVAL == 0 then
      --print("Memory: ", collectgarbage("count") * 1024)
      collectgarbage()
    end
  until coroutine.status(fetch_word) == 'dead'
  print("Vocabulary finished") 
end)




function make_alliterate(char)
  return coroutine.create(function(prev_char)
    local first_t = {}
    first_t[model.token_to_idx[char]] = 1
    first_t[model.token_to_idx[char:upper()]] = 1
    first_t[model.token_to_idx[' ']] = 1
    local weight_t = matches_to_weights(first_t)
    local weights = weight_t
    while true
      do
        p = coroutine.yield(weights)
        local next_char = model.idx_to_token[p[{1,1}]]
        if next_char:match("%W") then
          weights = weight_t
        else
          weights = {}
        end
      end
  end)
end

function make_suppressor(forbid)
  return coroutine.create(function(prev_char)
    local weights = {}
    for _, idx in pairs(model.token_to_idx) do
      weights[idx] = 1
    end
    print("Forbidden: ", forbid)
    for i = 1, #forbid do
      local c = forbid:sub(i,i)
      local idx = model.token_to_idx[c]
      if idx ~= nil then
        weights[idx] =0
      else
        print("unknown character", c)
      end
    end
    while true
      do
        p = coroutine.yield(weights)
      end
  end)
end




--print("Memory: ", collectgarbage("count") * 1024)

collectgarbage()

print(opt)

local sample = nil

if opt.suppress then
  vocab_filter = function(w)
    if utf8.match(w, '[' .. opt.suppress .. ']') then
      return true
    else
      return false
    end
  end
end



if opt.vocab == '' then
  if opt.alliterate ~= '' then
    print("Running in alliterate mode")
    local mod = make_alliterate(opt.alliterate:sub(1,1))
    sample = model:sample_hacked(opt, mod)
  else
    if opt.suppress ~= '' then
      print("Running in simple oulipo mode")
      local mod = make_suppressor(opt.suppress)
      sample = model:sample_hacked(opt, mod)
    else
      sample = model:sample(opt)
    end
  end
else
  if opt.excavate ~= 0 then
    print("Running in excavate mode")
    MAX_AHEAD = opt.excavate
    local tuner = make_vocab(excavate_vocab, vocab_filter)
    sample = model:sample_hacked(opt, tuner)
  else
    print("Running in whole-vocab mode")
    local tuner = make_vocab(basic_vocab, vocab_filter)
    sample = model:sample_hacked(opt, tuner)
  end
end


--print("End sampling:", collectgarbage("count") * 1024)

collectgarbage()

--print("After collection:", collectgarbage("count") * 1024)

opt['wordcount'] = #word_indices

local jsonfilename = opt.outdir .. '/' .. opt.name .. '.json'
local textfilename = opt.outdir .. '/' .. opt.name .. '.txt'

local jsonfile = io.open(jsonfilename, "w")
jsonfile:write(pretty({ words = word_indices, settings = opt }))
jsonfile:close()

print("Wrote word indices to " .. jsonfilename)

local textfile = io.open(textfilename, "w")
textfile:write(sample)
textfile:close()

print("Wrote text output to " .. textfilename)

print(sample)
